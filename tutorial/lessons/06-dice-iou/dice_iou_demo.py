"""
第 6 課：Dice Coefficient 與 IoU
資料集：MNIST（沿用第 5 課的『筆畫 = 前景』玩具分割任務）

這一課做兩件事：
  1) 用兩個簡單例子驗證 dice_score() / iou_score() 的計算是否正確，並確認
     Dice = 2*IoU / (1+IoU) 這個公式關係。
  2) 用一個「最陽春」的分割方法（固定灰階閾值）去分割 MNIST 數字，計算出
     來的 Dice / IoU，當作第 17 課「真的訓練一個 U-Net」的比較基準線。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def dice_score(pred_mask, gt_mask, eps=1e-7):
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


def sanity_check():
    print("== 手算範例，驗證公式 ==")
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[2:6, 2:6] = 1  # 4x4=16 pixels
    pred = np.zeros((10, 10), dtype=np.uint8)
    pred[3:7, 3:7] = 1  # 4x4=16 pixels, 和 gt 重疊 3x3=9 pixels

    d = dice_score(pred, gt)
    i = iou_score(pred, gt)
    print(f"Ground truth 面積=16, Prediction 面積=16, 交集=9")
    print(f"Dice = {d:.4f}   IoU = {i:.4f}")
    print(f"驗證 Dice = 2*IoU/(1+IoU) = {2 * i / (1 + i):.4f}  (應該要等於上面算出的 Dice)\n")


def threshold_baseline_on_mnist(n_samples=300, pred_threshold=100, gt_threshold=128):
    """用一個『比較寬鬆的閾值』模擬一個不太準的分割模型，
    和『比較嚴謹的閾值』當作 ground truth，計算平均 Dice / IoU。"""
    dataset = MNIST(root=str(DATA_DIR), train=True, download=True)
    imgs = dataset.data.numpy()[:n_samples]

    dices, ious = [], []
    for img in imgs:
        gt_mask = (img > gt_threshold).astype(np.uint8)
        pred_mask = (img > pred_threshold).astype(np.uint8)  # 閾值較低 -> 前景範圍偏大
        if gt_mask.sum() == 0:
            continue
        dices.append(dice_score(pred_mask, gt_mask))
        ious.append(iou_score(pred_mask, gt_mask))

    return np.array(dices), np.array(ious)


def main():
    sanity_check()

    print("== 用固定閾值分割 MNIST 數字，當作『陽春基準線』 ==")
    dices, ious = threshold_baseline_on_mnist()
    print(f"平均 Dice = {dices.mean():.4f} ± {dices.std():.4f}")
    print(f"平均 IoU  = {ious.mean():.4f} ± {ious.std():.4f}")
    print("=> 這條『陽春基準線』只是用另一個閾值去逼近正確的前景範圍，")
    print("   本身沒有『學習』能力。第 17 課會訓練一個真正的 U-Net 來做同一件事，")
    print("   到時候可以直接拿這裡的 Dice / IoU 數字來比較進步了多少。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 6 課）：
# 1) 把 pred_threshold 從 100 調到 150、200，觀察 Dice/IoU 如何隨著閾值
#    越來越接近 gt_threshold（128）而上升，這說明了「分割模型的品質」
#    如何直接反映在 Dice/IoU 上。
# 2) 用 dice_score / iou_score 這兩個函式，計算「prediction 完全等於
#    ground truth」與「prediction 完全沒有交集」兩種極端情況下的數值，
#    驗證是否符合你的預期（應該分別是 1.0 和 0.0）。
# ------------------------------------------------------------------
