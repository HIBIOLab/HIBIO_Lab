"""
第 5、6 課合併練習：在真實的 3D-IRCADb-01 資料集上訓練一個 Attention U-Net，
訓練完成後同時計算 accuracy / Dice / IoU，親眼比較這三個指標講的是不是同一件事。

資料集：3D-IRCADb-01（https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/）
        20 位病人的腹部 CT，含肝臟與肝腫瘤的人工標註。

前置作業：
  1) 到上面的網址免費註冊、下載資料集，解壓縮到 tutorial/data/3Dircadb1/
  2) 執行同資料夾的 prepare_ircadb_dataset.py，把原始 DICOM 轉成訓練用的切片
  3) 再執行這支程式

模型架構（Attention U-Net）與訓練流程參考自一份公開的 Kaggle notebook
（Attention U-Net 訓練同一個資料集，作者在 20 個 epoch 內把 tumor Dice 從
0.004 訓練到 0.87），這裡把它重新實作、簡化成教學用版本：
  - 影像縮小到 128x128（原本 256x256），epoch 數減少，讓一般筆電也能在
    合理時間內跑完
  - 拿掉一些工程細節（例如過度複雜的早停/警告訊息），保留核心結構：
    Attention Gate、Dice+BCE 混合 loss、liver+tumor 雙通道輸出
  - 每個 epoch 同時算出 accuracy 跟 Dice/IoU，訓練結束後跟『完全不訓練、
    永遠猜沒有』的全黑基準線並排比較，這就是第 5、6 課合併練習的重點
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SLICES_DIR = DATA_DIR / "ircadb_slices"      # prepare_ircadb_dataset.py 的輸出
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 8


def dice_score(pred_mask, gt_mask, eps=1e-7):
    """單張切片的 Dice（對照公式 7）。注意 eps 慣例：當預測與答案『都是空』
    (pred.sum()=gt.sum()=0) 時，會得到 eps/eps=1。這在『整張圖只有一個答案』
    時沒問題，但在分割裡，腫瘤大多數切片根本不存在，若逐張算再平均，這些空切片
    會把全黑模型的 Dice 灌到很高——所以評估請改用 evaluate() 的 dataset-level 版。"""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return (2 * intersection + eps) / (pred.sum() + gt.sum() + eps)


def iou_score(pred_mask, gt_mask, eps=1e-7):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (intersection + eps) / (union + eps)


def pixel_accuracy(pred_mask, gt_mask):
    return (pred_mask.astype(bool) == gt_mask.astype(bool)).mean()


# ------------------------------------------------------------------
# 資料集：讀取 prepare_ircadb_dataset.py 轉好的 .npz 切片
# ------------------------------------------------------------------

class IRCADbSliceDataset(Dataset):
    def __init__(self, npz_files, image_size=IMAGE_SIZE):
        self.files = npz_files
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def _resize(self, arr):
        h, w = arr.shape
        size = self.image_size
        if (h, w) == (size, size):
            return arr
        row_idx = (np.arange(size) * h / size).astype(int).clip(0, h - 1)
        col_idx = (np.arange(size) * w / size).astype(int).clip(0, w - 1)
        return arr[row_idx][:, col_idx]

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        img = self._resize(d["image"]).astype(np.float32)
        liver = self._resize(d["liver"]).astype(np.float32)
        tumor = self._resize(d["tumor"]).astype(np.float32)

        img_t = torch.from_numpy(img).unsqueeze(0)               # (1, H, W)
        mask_t = torch.stack([torch.from_numpy(liver), torch.from_numpy(tumor)])  # (2, H, W)
        return img_t, mask_t


def list_slice_files():
    if not SLICES_DIR.exists():
        return []
    return sorted(SLICES_DIR.glob("*.npz"))


def make_balanced_sampler(files):
    """跟參考 notebook 一樣：含腫瘤的切片給更高的抽樣權重，避免訓練時
    幾乎都是『沒有腫瘤』的切片，模型學不到腫瘤長怎樣。"""
    weights = []
    for f in files:
        d = np.load(f)
        has_tumor = d["tumor"].sum() > 0
        weights.append(3.0 if has_tumor else 1.0)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ------------------------------------------------------------------
# Attention U-Net
# ------------------------------------------------------------------

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class AttentionGate(nn.Module):
    """Attention Gate：讓 decoder 在做 skip connection 前，先學會『該看
    encoder 特徵圖的哪個區域』，而不是照單全收。W_g 處理來自 decoder（比較
    抽象、語意層級高）的訊號，W_x 處理來自 encoder（比較細節）的訊號，
    兩者相加、經過 ReLU 和 1x1 conv + sigmoid，得到一張 0~1 的注意力權重圖
    psi，最後把 encoder 特徵乘上 psi，等於『放大』模型該注意的區域。"""

    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.w_g(g) + self.w_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, out_channels=2):
        super().__init__()
        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(128, 128, 64)
        self.dec3 = conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(64, 64, 32)
        self.dec2 = conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(32, 32, 16)
        self.dec1 = conv_block(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.upconv3(b)
        e3_att = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.upconv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.upconv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        return torch.sigmoid(self.out_conv(d1))


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.7, bce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def dice_loss(self, pred, target):
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

    def forward(self, pred, target):
        # liver（通道 0）跟 tumor（通道 1）分開算，tumor 稀少所以給更高權重
        liver_loss = (self.dice_weight * self.dice_loss(pred[:, 0], target[:, 0])
                      + self.bce_weight * self.bce(pred[:, 0], target[:, 0]))
        tumor_loss = (self.dice_weight * self.dice_loss(pred[:, 1], target[:, 1])
                      + self.bce_weight * self.bce(pred[:, 1], target[:, 1]))
        return liver_loss + 2.0 * tumor_loss


CHANNELS = ["liver", "tumor"]


def evaluate(model, loader):
    """在驗證集上算出 liver / tumor 兩個通道的 accuracy、Dice、IoU。

    ⚠️ 重點：Dice / IoU 用『整個驗證集一起累加』(dataset-level) 的方式算，
    不是每張切片各算一次 Dice 再平均。原因是——腫瘤在大多數切片裡根本不存在，
    如果每張切片各算一次，那些『答案是空、預測也是空』的切片會被 dice_score 的
    eps 慣例算成 Dice=1（0/0→1）。一旦把這些切片一起平均，連『全黑什麼都不猜』
    的模型 tumor Dice 都會虛高到 0.6，甚至比真的有在學的模型還高，完全誤導。
    改成整個資料集累加交集/聯集後，全黑模型的 Dice 就會正確地趨近 0。
    （dice_score / iou_score 這兩個 per-slice 版本保留下來對照公式 6、7 用。）"""
    model.eval()
    agg = {ch: {"inter": 0, "union": 0, "pred": 0, "gt": 0, "correct": 0, "total": 0}
           for ch in CHANNELS}
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = (model(x) > 0.3).cpu().numpy().astype(bool)
            y = y.numpy().astype(bool)
            for ci, ch in enumerate(CHANNELS):
                p, g = pred[:, ci], y[:, ci]
                a = agg[ch]
                a["inter"] += int(np.logical_and(p, g).sum())
                a["union"] += int(np.logical_or(p, g).sum())
                a["pred"] += int(p.sum())
                a["gt"] += int(g.sum())
                a["correct"] += int((p == g).sum())
                a["total"] += p.size
    out = {}
    for ch in CHANNELS:
        a = agg[ch]
        out[f"{ch}_acc"] = a["correct"] / a["total"]
        out[f"{ch}_dice"] = 2 * a["inter"] / (a["pred"] + a["gt"] + 1e-7)
        out[f"{ch}_iou"] = a["inter"] / (a["union"] + 1e-7)
    return out


def naive_all_black_baseline(loader):
    """完全不訓練、永遠預測『沒有腫瘤/沒有肝臟』的全黑基準線。
    一樣用 dataset-level 累加：全黑 → 交集=0、預測像素=0，所以 Dice = 0；
    但 accuracy 依然很高（背景像素佔絕大多數）——這正是這一課要凸顯的重點。"""
    agg = {ch: {"gt": 0, "correct": 0, "total": 0} for ch in CHANNELS}
    for _, y in loader:
        y = y.numpy().astype(bool)
        for ci, ch in enumerate(CHANNELS):
            g = y[:, ci]
            a = agg[ch]
            a["gt"] += int(g.sum())
            a["correct"] += int((~g).sum())   # 全黑=False，猜對的就是背景像素
            a["total"] += g.size
    out = {}
    for ch in CHANNELS:
        a = agg[ch]
        out[f"{ch}_acc"] = a["correct"] / a["total"]
        out[f"{ch}_dice"] = 0.0   # 交集=0、預測像素=0 → dice 恆為 0
    return out


def main():
    print(f"使用裝置: {DEVICE}")
    files = list_slice_files()
    if not files:
        print(f"\n找不到切片資料: {SLICES_DIR}")
        print("請先完成前置作業：")
        print("  1) 到 https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/")
        print("     免費註冊、下載 3D-IRCADb-01，解壓縮到 tutorial/data/3Dircadb1/")
        print("  2) 執行 python prepare_ircadb_dataset.py 轉換成訓練用切片")
        print("  3) 重新執行這支程式")
        return

    n_val = max(1, int(len(files) * 0.2))
    train_files, val_files = files[:-n_val], files[-n_val:]
    print(f"切片總數: {len(files)}（train={len(train_files)}, val={len(val_files)}）")

    train_ds = IRCADbSliceDataset(train_files)
    val_ds = IRCADbSliceDataset(val_files)
    sampler = make_balanced_sampler(train_files)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionUNet(out_channels=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    loss_fn = DiceBCELoss()

    keys = ["liver_acc", "tumor_acc", "liver_dice", "tumor_dice", "liver_iou", "tumor_iou"]
    history = {"train_loss": [], **{k: [] for k in keys}}

    print(f"\n開始訓練 Attention U-Net（{EPOCHS} epochs）...")
    print("每個 epoch 都會同時印出 accuracy 跟 Dice，留意兩者何時對不上：")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_ds)
        metrics = evaluate(model, val_loader)
        history["train_loss"].append(train_loss)
        for k in keys:
            history[k].append(metrics[k])
        print(f"  epoch {epoch+1:2d}/{EPOCHS}  loss={train_loss:.4f}  |  "
              f"tumor: acc={metrics['tumor_acc']:.4f} dice={metrics['tumor_dice']:.4f}  |  "
              f"liver: acc={metrics['liver_acc']:.4f} dice={metrics['liver_dice']:.4f}")

    # ------------------------------------------------------------------
    # 訓練完成：跟『完全不訓練、永遠猜沒有』的基準線並排比較 accuracy / Dice
    # ------------------------------------------------------------------
    baseline = naive_all_black_baseline(val_loader)
    final = {k: history[k][-1] for k in keys}

    print("\n" + "=" * 62)
    print("== 訓練完成，比較「全黑基準線」 vs 「訓練好的 Attention U-Net」 ==")
    print("=" * 62)
    print("（Dice / IoU 為 dataset-level：整個驗證集一起累加交集/聯集後計算）")
    print(f"{'':22s} {'accuracy':>10s} {'dice':>10s} {'iou':>10s}")
    print(f"{'全黑基準線 - liver':22s} {baseline['liver_acc']:10.4f} {baseline['liver_dice']:10.4f} {'--':>10s}")
    print(f"{'全黑基準線 - tumor':22s} {baseline['tumor_acc']:10.4f} {baseline['tumor_dice']:10.4f} {'--':>10s}")
    print(f"{'Attention U-Net - liver':22s} {final['liver_acc']:10.4f} {final['liver_dice']:10.4f} {final['liver_iou']:10.4f}")
    print(f"{'Attention U-Net - tumor':22s} {final['tumor_acc']:10.4f} {final['tumor_dice']:10.4f} {final['tumor_iou']:10.4f}")
    print("\n=> 全黑基準線的 accuracy 通常已經很高（腫瘤像素本來就稀少），但 Dice")
    print("   幾乎是 0；訓練好的模型則是 accuracy 和 Dice 應該要『一起』變高。")
    print("   如果你看到 accuracy 很高但 Dice 還是很低，代表模型可能還是傾向")
    print("   『多猜背景』討好 accuracy，還沒真的學會找出腫瘤——這正是這兩課")
    print("   合併練習要你動手驗證的重點。")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(history["tumor_acc"], label="Tumor Accuracy")
    axes[1].plot(history["tumor_dice"], label="Tumor Dice")
    axes[1].axhline(baseline["tumor_acc"], color="gray", linestyle="--", label="全黑基準線 Accuracy")
    axes[1].set_title("Tumor: Accuracy vs Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    axes[2].plot(history["liver_acc"], label="Liver Accuracy")
    axes[2].plot(history["liver_dice"], label="Liver Dice")
    axes[2].set_title("Liver: Accuracy vs Dice")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "06_attention_unet_training.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n訓練曲線已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（第 5、6 課合併練習，對照 index.html）：
#
# 1) accuracy 與 Dice 何時「兜不起來」＋ 評估指標本身也會騙人：
#    對照最後的「全黑基準線 vs Attention U-Net」表格——全黑基準線的 tumor
#    accuracy 高達 0.99，Dice 卻是 0.0，為什麼 accuracy 會這麼高？
#    進階：本課的 Dice 是用「整個驗證集一起累加」(dataset-level) 算的。如果
#    改成「每張切片各算一次 Dice 再平均」，全黑基準線的 tumor Dice 會從 0
#    跳到約 0.6。想想為什麼？（提示：一張「沒有腫瘤」的切片，全黑預測的
#    Dice 是 0/0，該算成 0 還是 1？見 dice_score 的 eps 慣例）——這說明連
#    「指標的聚合方式」都可能製造假象。
#
# 2) 為什麼 tumor 遠比 liver 難？跑完你會看到 liver Dice 爬到 ~0.87，tumor
#    Dice 卻卡在 ~0.1 上不去。已知 tumor 只占約 0.29% 的像素、比 liver 稀有
#    約 17 倍，且四分之一的腫瘤不到 100 個像素。用「類別不平衡 + 小物件」
#    解釋為什麼 tumor 這麼難，以及為什麼「幾乎全猜背景」在 tumor 上仍能拿到
#    0.99 accuracy。（註：驗證集的腫瘤其實偏大也沒被 128px 縮沒，所以主因是
#    模型沒學好 tumor，而不是解析度把它弄丟。）
#
# 3) 從「loss」把 tumor 拉上去：DiceBCELoss 裡 tumor_loss 乘了 2.0、liver
#    沒有，為什麼？把這個權重拿掉重訓，你預期 tumor Dice 變好還是變差？
#    進階：查一下 Tversky / Focal-Tversky loss，說明它為什麼比 Dice+BCE 更
#    適合小病灶（提示：它讓「漏抓 FN」的懲罰大於「誤報 FP」）。
#
# 4) 從「推論/訓練設定」把 tumor 拉上去：目前預測用 model(x) > 0.3。tumor 的
#    機率普遍偏低，試著只把 tumor 的閾值降到 0.15~0.2（liver 維持 0.3）重新
#    評估，tumor Dice 有沒有回升？這對應到 precision/recall 的什麼取捨？再
#    試著把 EPOCHS 調到 40~60、或 IMAGE_SIZE 調回 256，比較哪個改動對 tumor
#    Dice 影響最大、能不能往參考 notebook 的 0.87 靠近。
#
# 5) AttentionGate.forward() 裡 x * psi 這一步在做什麼？psi 的數值範圍是多少？
#    對照 index.html 的圖解，找出 w_g、w_x、psi 分別對應圖中的哪一個步驟。
# ------------------------------------------------------------------
