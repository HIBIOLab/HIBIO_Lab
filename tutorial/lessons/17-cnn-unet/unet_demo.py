"""
第 17 課 (2/2)：U-Net —— 醫學影像分割的標配架構
資料集：Retina Blood Vessel（視網膜眼底影像的血管分割）
        放在 tutorial/data/Retina blood/ 底下（train/ 與 test/，各有 image/ 與 mask/）
        來源：https://www.kaggle.com/datasets/abdallahwagih/retina-blood-vessel

這裡訓練一個真正的 U-Net（encoder-decoder + skip connection）做『血管分割』
（每個像素判斷是不是血管），並用 dice_score / iou_score（公式跟第 14 課相同）
算出分數，跟一個『陽春的綠色通道閾值基準線』比較 —— 讓你看到真正訓練出來的
模型，比手工設計的規則好在哪裡。

備註：眼底影像原始大小 512x512，為了讓沒有 GPU 也能在數分鐘內跑完，程式
預設把影像縮到 128x128、用較小的 U-Net。想要更漂亮的血管細節，可以把
IMG_SIZE 調到 256 或 512、並在有 GPU 的機器上跑。
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RETINA_DIR = DATA_DIR / "Retina blood"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 128          # 影像縮放後的邊長（想要更細的血管可調到 256 / 512）
MASK_THRESHOLD = 0.5    # 遮罩像素 > 0.5 視為血管（前景）


# ---------- 評估指標（跟第 14 課相同）----------
def dice_score(pred_mask, gt_mask, eps=1e-7):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def iou_score(pred_mask, gt_mask, eps=1e-7):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + eps) / (union + eps)


# ---------- U-Net 架構（encoder-decoder + skip connection）----------
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """3 次下採樣的 U-Net（RGB 3 通道進、1 通道血管機率出）。"""

    def __init__(self, base=16):
        super().__init__()
        self.enc1 = conv_block(3, base)              # 128
        self.enc2 = conv_block(base, base * 2)       # 64
        self.enc3 = conv_block(base * 2, base * 4)   # 32
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base * 4, base * 8)  # 16

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # skip connection
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # skip connection
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # skip connection
        return self.out_conv(d1)  # logits，未經 sigmoid


class DiceBCELoss(nn.Module):
    """Dice loss + BCE，血管分割這種前景很細的任務常用的組合（見參考 notebook）。"""

    def forward(self, logits, targets, smooth=1.0):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        dice_loss = 1 - (2 * inter + smooth) / (probs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(probs, targets)
        return bce + dice_loss


# ---------- 資料載入 ----------
def load_split(split):
    """讀取 Retina blood/<split>/image 與 mask，回傳 (X, Y) 兩個 tensor。"""
    img_dir = RETINA_DIR / split / "image"
    mask_dir = RETINA_DIR / split / "mask"
    if not img_dir.exists():
        raise FileNotFoundError(
            f"找不到 {img_dir}。請確認 Retina blood 資料集已放在 {RETINA_DIR}（含 train/ 與 test/）。")

    stems = sorted((p.stem for p in img_dir.glob("*.png")), key=lambda s: int(s) if s.isdigit() else s)
    imgs, masks = [], []
    for stem in stems:
        img = Image.open(img_dir / f"{stem}.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        msk = Image.open(mask_dir / f"{stem}.png").convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        imgs.append(np.asarray(img, dtype=np.float32) / 255.0)
        masks.append((np.asarray(msk, dtype=np.float32) / 255.0 > MASK_THRESHOLD).astype(np.float32))

    X = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2)   # (N,3,H,W)
    Y = torch.from_numpy(np.stack(masks)).unsqueeze(1)         # (N,1,H,W)
    return X, Y


def green_threshold_baseline(X, Y):
    """陽春基準線：血管在眼底影像的『綠色通道』最暗，取視野內最暗的一群像素當血管。"""
    dices, ious = [], []
    imgs = X.numpy()
    gts = Y.numpy()[:, 0]
    for img, gt in zip(imgs, gts):
        green = img[1]                      # 綠色通道
        lum = img.mean(axis=0)              # 亮度，用來抓出圓形視野 (FOV)
        fov = lum > 0.1
        target_ratio = gt.mean()            # 用真實血管比例當作要挑多暗的門檻
        if fov.sum() == 0 or target_ratio <= 0:
            pred = np.zeros_like(green, dtype=np.uint8)
        else:
            thr = np.quantile(green[fov], target_ratio)
            pred = ((green <= thr) & fov).astype(np.uint8)
        dices.append(dice_score(pred, gt))
        ious.append(iou_score(pred, gt))
    return np.array(dices), np.array(ious)


def main():
    print(f"使用裝置: {DEVICE}  影像大小: {IMG_SIZE}x{IMG_SIZE}")
    X_train, Y_train = load_split("train")
    X_test, Y_test = load_split("test")
    print(f"訓練集 {len(X_train)} 張，測試集 {len(X_test)} 張\n")

    print("== 陽春基準線（綠色通道閾值分割，當作跟 U-Net 比較的對照）==")
    base_dices, base_ious = green_threshold_baseline(X_test, Y_test)
    print(f"平均 Dice = {base_dices.mean():.4f}   平均 IoU = {base_ious.mean():.4f}\n")

    print("== 訓練 U-Net ==")
    torch.manual_seed(0)
    model = UNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"參數量: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DiceBCELoss()

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=4, shuffle=True)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"  epoch {epoch+1:2d}/{epochs}  train_loss={total_loss/len(X_train):.4f}")

    print("\n== 在測試集上計算 U-Net 的 Dice / IoU ==")
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(X_test)):
            logits = model(X_test[i:i + 1].to(DEVICE))
            preds.append((torch.sigmoid(logits).cpu().numpy()[0, 0] > 0.5).astype(np.uint8))
    preds = np.stack(preds)
    gts = Y_test.numpy()[:, 0]

    unet_dices = np.array([dice_score(p, g) for p, g in zip(preds, gts)])
    unet_ious = np.array([iou_score(p, g) for p, g in zip(preds, gts)])
    print(f"U-Net 平均 Dice = {unet_dices.mean():.4f}   平均 IoU = {unet_ious.mean():.4f}")

    print(f"\n== 總結比較（陽春綠色通道基準線 vs 這裡訓練出來的 U-Net）==")
    print(f"  陽春閾值基準線: Dice={base_dices.mean():.4f}  IoU={base_ious.mean():.4f}")
    print(f"  U-Net         : Dice={unet_dices.mean():.4f}  IoU={unet_ious.mean():.4f}")
    print("=> U-Net 靠encoder 抓上下文、skip connection 把細節接回來，通常能作為basleine。")

    # 視覺化幾張範例
    n_show = min(5, len(X_test))
    fig, axes = plt.subplots(3, n_show, figsize=(2.4 * n_show, 7))
    for i in range(n_show):
        axes[0, i].imshow(X_test[i].permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"眼底影像 #{i}")
        axes[1, i].imshow(gts[i], cmap="gray")
        axes[1, i].set_title("Ground Truth")
        axes[2, i].imshow(preds[i], cmap="gray")
        axes[2, i].set_title(f"U-Net 預測\nDice={unet_dices[i]:.3f}")
        for row in range(3):
            axes[row, i].axis("off")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "17_unet_retina.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 17 課）：
# 1) U-Net 的 skip connection（forward 裡 torch.cat([...], dim=1) 那幾行）
#    作用是什麼？試著把 skip connection 拿掉（改成只用 up 的輸出、不 concat，
#    並把對應 decoder 的輸入通道數改成一半），重新訓練，比較 Dice/IoU 是否
#    下降？血管這種細結構特別依賴 skip connection 傳回來的高解析度細節。
# 2) 把 IMG_SIZE 從 128 調到 256，血管細節變清楚後 Dice/IoU 是否提升？
#    訓練時間又增加多少？（這說明了解析度與計算成本之間的取捨。）
# ------------------------------------------------------------------
