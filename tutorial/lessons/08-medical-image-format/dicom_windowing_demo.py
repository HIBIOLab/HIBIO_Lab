"""
第 8 課：醫學影像資料格式與前處理（DICOM / HU / Windowing）
資料集：pydicom 套件內建的真實 CT DICOM 範例檔（CT_small.dcm），
        不需要另外下載，pip install pydicom 就會一起裝好。

這一課示範：
  1) 讀取 DICOM，看看裡面除了影像還有哪些 metadata（示範為何需要去識別化）。
  2) 用 RescaleSlope / RescaleIntercept 把原始像素值換算成 Hounsfield Unit (HU)。
  3) 用不同的 window level / width 產生「同一張 CT，不同對比呈現」的圖片，
     具體看到 windowing 對可視化的影響。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pydicom.data import get_testdata_file

# Windows 預設字型不含中文字形，圖片標題若有中文需指定支援的字型才不會變成空格
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 常見的臨床 window 設定 (window_level, window_width)，單位為 HU
WINDOW_PRESETS = {
    "soft_tissue（軟組織）": (40, 400),
    "lung（肺窗，範圍很寬）": (-600, 1500),
    "bone（骨窗）": (400, 1800),
}


def load_dicom():
    path = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(path)
    return ds


def apply_window(hu_array, level, width):
    """把 HU 值依 window level/width 線性映射到 0-255，超出範圍的直接裁切。"""
    low = level - width / 2
    high = level + width / 2
    windowed = np.clip(hu_array, low, high)
    windowed = (windowed - low) / (high - low) * 255.0
    return windowed.astype(np.uint8)


def main():
    ds = load_dicom()

    print("== DICOM Metadata（節錄，示範隱私相關欄位）==")
    print(f"Modality (影像類型): {ds.get('Modality', 'N/A')}")
    print(f"影像大小: {ds.Rows} x {ds.Columns}")
    print(f"Patient Name 欄位: {ds.get('PatientName', 'N/A')}  <- 真實資料這裡會是病人姓名，")
    print("   使用前必須去識別化 (anonymize)，不能直接把原始 DICOM 分享出去。")

    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    raw = ds.pixel_array.astype(np.float32)
    hu = raw * slope + intercept

    print(f"\n== Hounsfield Unit 換算 ==")
    print(f"RescaleSlope={slope}, RescaleIntercept={intercept}")
    print(f"原始像素值範圍: [{raw.min():.0f}, {raw.max():.0f}]")
    print(f"換算成 HU 後範圍: [{hu.min():.0f}, {hu.max():.0f}]  "
          f"(水的 HU 應該接近 0，空氣接近 -1000，骨頭通常 > 400)")

    print("\n== 套用不同 window 設定 ==")
    fig, axes = plt.subplots(1, len(WINDOW_PRESETS) + 1, figsize=(4 * (len(WINDOW_PRESETS) + 1), 4))
    axes[0].imshow(hu, cmap="gray")
    axes[0].set_title("原始 HU（未 windowing）")
    axes[0].axis("off")

    for ax, (name, (level, width)) in zip(axes[1:], WINDOW_PRESETS.items()):
        windowed = apply_window(hu, level, width)
        ax.imshow(windowed, cmap="gray")
        ax.set_title(f"{name}\nlevel={level}, width={width}")
        ax.axis("off")
        print(f"  {name}: level={level}, width={width} "
              f"-> 顯示範圍 HU [{level - width / 2:.0f}, {level + width / 2:.0f}]")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "08_dicom_windowing.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")
    print("=> 同一張 CT，window 設定不同，看起來對比完全不一樣。")
    print("   如果不做 windowing，直接把 16-bit HU 線性壓成灰階，畫面通常會")
    print("   很平、看不出細節，因為動態範圍被稀釋掉了。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 8 課）：
# 1) 自己調整 WINDOW_PRESETS 的 level/width，觀察哪個設定最容易看出影像
#    中對比最高的區域（提示：width 越窄，對比越強但可視範圍越小）。
# 2) 寫一個函式，計算「如果完全不做 windowing，直接把 HU 線性壓縮到
#    0-255」的圖片，和 soft_tissue window 的結果放在一起比較視覺差異。
# ------------------------------------------------------------------
