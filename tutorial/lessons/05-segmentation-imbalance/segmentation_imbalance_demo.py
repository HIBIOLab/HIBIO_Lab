"""
第 5 課：分割任務導論與背景類別不平衡
資料集：
  (A) MNIST 手寫數字 —— 把每張圖二值化，筆畫像素當作『前景』，其餘當『背景』，
      拿來當作最簡單的玩具分割任務（後面第 6、17 課會延續使用同一份資料）。
  (B) 用 numpy 產生的合成影像 —— 128x128 的黑色背景中放一個 6x6 的小『腫瘤』，
      更貼近真實醫學影像中『病灶只佔一小塊』的極端比例。

目的：讓你具體看到「背景佔多數」對 pixel accuracy 這個指標有多大的影響。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def mnist_foreground_ratio(n_samples=500):
    dataset = MNIST(root=str(DATA_DIR), train=True, download=True)
    imgs = dataset.data.numpy()[:n_samples]  # uint8, 像素值範圍 0-255
    masks = (imgs > 128).astype(np.uint8)  # 筆畫 = 前景 (1)
    ratios = masks.reshape(n_samples, -1).mean(axis=1)
    return ratios


def naive_all_background_accuracy(foreground_ratio):
    """如果模型永遠只預測『背景』，pixel accuracy 就等於背景像素的比例。"""
    return 1 - foreground_ratio


def synthetic_tiny_tumor(image_size=128, tumor_size=6):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cy, cx = image_size // 2, image_size // 2
    half = tumor_size // 2
    mask[cy - half: cy + half, cx - half: cx + half] = 1
    return mask


def main():
    print("== (A) MNIST 手寫數字當作玩具分割任務 ==")
    ratios = mnist_foreground_ratio()
    print(f"500 張圖的平均前景（筆畫）像素比例: {ratios.mean():.2%}")
    naive_acc = naive_all_background_accuracy(ratios.mean())
    print(f"若模型『永遠只預測背景』，pixel accuracy = {naive_acc:.2%}")
    print("=> 即使 MNIST 的前景比例不算太小，光靠『猜背景』就能拿到八成多的")
    print("   pixel accuracy，這個指標本身無法告訴你模型有沒有真的分割出數字。\n")

    print("== (B) 合成影像：128x128 背景中只有一顆 6x6 的『腫瘤』 ==")
    mask = synthetic_tiny_tumor()
    total_pixels = mask.size
    tumor_pixels = mask.sum()
    foreground_ratio = tumor_pixels / total_pixels
    print(f"腫瘤像素數: {tumor_pixels} / 總像素數: {total_pixels}"
          f"（前景比例 = {foreground_ratio:.3%}）")
    naive_acc_tumor = 1 - foreground_ratio
    print(f"若模型『永遠只預測背景』，pixel accuracy = {naive_acc_tumor:.3%}")
    print("=> 這才是醫學影像分割常見的真實情況：一個『什麼都沒偵測到』的模型")
    print(f"   pixel accuracy 可以高達 {naive_acc_tumor:.2%}，但臨床上完全沒用。")
    print("   這就是為什麼分割任務一定要看 Dice / IoU（下一課），而不能只看")
    print("   pixel accuracy。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 5 課）：
# 1) 把 synthetic_tiny_tumor 的 tumor_size 改成 20，重新計算前景比例與
#    naive pixel accuracy，觀察病灶變大後，這個陷阱有沒有變得比較不明顯？
# 2) 用你自己的話寫一段話，解釋為什麼「pixel accuracy 很高」不能當作
#    「分割做得好」的證據，並舉出至少一個可以取代它的指標名稱
#    （提示：下一課會教）。
# ------------------------------------------------------------------
