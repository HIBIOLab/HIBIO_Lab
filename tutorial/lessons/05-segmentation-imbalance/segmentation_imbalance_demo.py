"""
第 5 課：分割任務導論與背景類別不平衡

這支程式用真實的 3D-IRCADb-01 肝臟腫瘤 CT 資料集（20 位病人），算出「整個
資料集」的腫瘤／背景像素比例，讓你具體看到「背景佔絕大多數」這件事，對
pixel accuracy 這個指標到底有多大的影響。

前置作業：先照第 6 課 prepare_ircadb_dataset.py 的說明下載、轉換好資料集，
再執行這支程式。還沒轉換好也沒關係，程式會印出清楚的指引而不會報錯中斷。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

SLICES_DIR = Path(__file__).resolve().parents[2] / "data" / "ircadb_slices"


def real_dataset_statistics():
    print("== 真實資料集：3D-IRCADb-01 整體的腫瘤／背景像素比例 ==")
    files = sorted(SLICES_DIR.glob("*.npz"))
    if not files:
        print(f"找不到轉換好的切片: {SLICES_DIR}")
        print("請先照第 6 課 prepare_ircadb_dataset.py 的說明下載、轉換好")
        print("3D-IRCADb-01，再重新執行這支程式，就會看到全資料集的真實統計數字。")
        return

    total_pixels = 0
    total_tumor_pixels = 0
    slices_with_tumor = 0
    for f in files:
        d = np.load(f)
        total_pixels += d["tumor"].size
        tumor_pixels = int(d["tumor"].sum())
        total_tumor_pixels += tumor_pixels
        if tumor_pixels > 0:
            slices_with_tumor += 1

    ratio = total_tumor_pixels / total_pixels
    print(f"切片總數: {len(files)}（來自多位病人）")
    print(f"有腫瘤的切片數: {slices_with_tumor} ({slices_with_tumor/len(files):.1%})")
    print(f"整個資料集的腫瘤像素比例: {ratio:.4%}")
    print(f"若模型『永遠只預測背景』，整體 pixel accuracy = {1-ratio:.4%}")
    print("=> 即使跨越整個資料集、上千張切片一起算，腫瘤像素依然只佔極小一部分：")
    print("   一個『什麼都沒偵測到』的模型，pixel accuracy 可以高達 99% 以上，臨床")
    print("   上卻完全沒用。這就是為什麼分割任務一定要看 Dice / IoU（第 6 課），")
    print("   而不能只看 pixel accuracy。")


def main():
    real_dataset_statistics()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習：這一課沒有獨立的課後練習，跟第 6 課合併在一起做
# （對照 index.html 第 6 課「Dice Coefficient 與 IoU」的課後練習）。
# ------------------------------------------------------------------
