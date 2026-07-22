"""
第 17 課 (2/2)：U-Net —— 醫學影像分割的標配架構
資料集：MNIST（沿用第 5、6 課「筆畫 = 前景」的玩具分割任務）

這裡訓練一個真正的迷你 U-Net（encoder-decoder + skip connection）來做
『分割』（每個像素判斷是不是筆畫），並用第 6 課定義的 dice_score /
iou_score 算出來的分數，直接跟第 6 課『陽春閾值基準線』比較。
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
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GT_THRESHOLD = 128  # 和第 6 課一致：像素值 > 128 視為前景 (筆畫)


def dice_score(pred_mask, gt_mask, eps=1e-7):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)


def iou_score(pred_mask, gt_mask, eps=1e-7):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (inter + eps) / (union + eps)


class TinyUNet(nn.Module):
    """兩層 encoder + 兩層 decoder 的迷你 U-Net，附 skip connection。"""

    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.enc1 = block(1, 16)          # 28x28
        self.pool1 = nn.MaxPool2d(2)      # -> 14x14
        self.enc2 = block(16, 32)         # 14x14
        self.pool2 = nn.MaxPool2d(2)      # -> 7x7

        self.bottleneck = block(32, 64)   # 7x7

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # 7x7 -> 14x14
        self.dec2 = block(64, 32)         # 32(up) + 32(skip) = 64 in

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 14x14 -> 28x28
        self.dec1 = block(32, 16)         # 16(up) + 16(skip) = 32 in

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # skip connection：把 encoder 的細節接回來

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # skip connection

        return self.out_conv(d1)  # logits，未經 sigmoid


def load_mnist_masks(n_train=2000, n_test=500):
    train_set = MNIST(root=str(DATA_DIR), train=True, download=True)
    test_set = MNIST(root=str(DATA_DIR), train=False, download=True)

    def to_tensors(dataset, n):
        imgs = dataset.data.numpy()[:n].astype(np.float32) / 255.0
        masks = (dataset.data.numpy()[:n] > GT_THRESHOLD).astype(np.float32)
        X = torch.from_numpy(imgs).unsqueeze(1)   # (N,1,28,28)
        Y = torch.from_numpy(masks).unsqueeze(1)  # (N,1,28,28)
        return X, Y

    X_train, Y_train = to_tensors(train_set, n_train)
    X_test, Y_test = to_tensors(test_set, n_test)
    return X_train, Y_train, X_test, Y_test


def naive_threshold_baseline(X_test, Y_test, pred_threshold=100 / 255):
    dices, ious = [], []
    imgs = X_test.numpy()[:, 0]
    gts = Y_test.numpy()[:, 0]
    for img, gt in zip(imgs, gts):
        pred = (img > pred_threshold).astype(np.uint8)
        dices.append(dice_score(pred, gt))
        ious.append(iou_score(pred, gt))
    return np.array(dices), np.array(ious)


def main():
    print(f"使用裝置: {DEVICE}")
    X_train, Y_train, X_test, Y_test = load_mnist_masks()
    print(f"訓練集: {len(X_train)} 張，測試集: {len(X_test)} 張\n")

    print("== 陽春基準線（第 6 課的固定閾值方法，這裡重新算一次當對照）==")
    base_dices, base_ious = naive_threshold_baseline(X_test, Y_test)
    print(f"平均 Dice = {base_dices.mean():.4f}   平均 IoU = {base_ious.mean():.4f}\n")

    print("== 訓練 TinyUNet ==")
    model = TinyUNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

    epochs = 8
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"  epoch {epoch+1}/{epochs}  train_loss={total_loss/len(X_train):.4f}")

    print("\n== 在測試集上計算 U-Net 的 Dice / IoU ==")
    model.eval()
    with torch.no_grad():
        logits = model(X_test.to(DEVICE))
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
    preds = (probs > 0.5).astype(np.uint8)
    gts = Y_test.numpy()[:, 0]

    unet_dices = np.array([dice_score(p, g) for p, g in zip(preds, gts)])
    unet_ious = np.array([iou_score(p, g) for p, g in zip(preds, gts)])
    print(f"U-Net 平均 Dice = {unet_dices.mean():.4f}   平均 IoU = {unet_ious.mean():.4f}")

    print(f"\n== 總結比較（第 6 課基準線 vs 這裡訓練出來的 U-Net）==")
    print(f"  陽春閾值基準線: Dice={base_dices.mean():.4f}  IoU={base_ious.mean():.4f}")
    print(f"  TinyUNet      : Dice={unet_dices.mean():.4f}  IoU={unet_ious.mean():.4f}")

    # 視覺化幾張範例
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    for i in range(6):
        axes[0, i].imshow(X_test[i, 0], cmap="gray")
        axes[0, i].set_title(f"輸入影像 #{i}")
        axes[1, i].imshow(gts[i], cmap="gray")
        axes[1, i].set_title("Ground Truth")
        axes[2, i].imshow(preds[i], cmap="gray")
        axes[2, i].set_title(f"U-Net 預測\nDice={unet_dices[i]:.3f}")
        for row in range(3):
            axes[row, i].axis("off")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "17_unet_segmentation.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 17 課）：
# 1) U-Net 的 skip connection（torch.cat([d2, e2], dim=1) 那幾行）作用
#    是什麼？試著把 skip connection 拿掉（改成不 concat，直接用 up 的
#    輸出），重新訓練，比較 Dice/IoU 是否下降？
# 2) 把訓練資料 n_train 從 2000 降到 200，重新比較 U-Net 跟陽春基準線的
#    差距，資料少的時候 U-Net 還能明顯勝過陽春方法嗎？這跟醫學影像資料
#    通常很少的現實情況有什麼關聯？
# ------------------------------------------------------------------
