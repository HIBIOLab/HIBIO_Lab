"""
第 6 課（互動版）：醫學影像資料格式與前處理（DICOM / HU / Windowing）
資料集：pydicom 套件內建的真實 CT DICOM 範例檔（CT_small.dcm），
        不需要另外下載，pip install pydicom 就會一起裝好。

"""

import sys

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import pydicom
from pydicom.data import get_testdata_file

# Windows 預設字型不含中文字形，標題若有中文需指定支援的字型才不會變成空格
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 常見的臨床 window 設定 (window_level, window_width)，單位為 HU
WINDOW_PRESETS = {
    "軟組織 soft": (40, 400),
    "肺窗 lung": (-600, 1500),
    "骨窗 bone": (400, 1800),
}

# slider 可調範圍（HU）
LEVEL_MIN, LEVEL_MAX = -1000, 1500
WIDTH_MIN, WIDTH_MAX = 1, 3000


def load_dicom():
    path = get_testdata_file("CT_small.dcm")
    return pydicom.dcmread(path)


def dicom_to_hu(ds):
    """用 RescaleSlope / RescaleIntercept 把原始像素值換算成 Hounsfield Unit。"""
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    raw = ds.pixel_array.astype(np.float32)
    return raw * slope + intercept, slope, intercept


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

    hu, slope, intercept = dicom_to_hu(ds)
    print("\n== Hounsfield Unit 換算 ==")
    print(f"RescaleSlope={slope}, RescaleIntercept={intercept}")
    print(f"換算成 HU 後範圍: [{hu.min():.0f}, {hu.max():.0f}]  "
          f"(水的 HU 應該接近 0，空氣接近 -1000，骨頭通常 > 400)")
    print("\n拉下方的 slider 調整 window level / width，影像會即時更新。")
    print("也可以按下方的預設按鈕，直接跳到常見的臨床 window。")

    # 初始值用軟組織 window
    init_level, init_width = WINDOW_PRESETS["軟組織 soft"]

    fig = plt.figure(figsize=(8, 8.5))

    # --- 影像區 ---
    ax_img = fig.add_axes([0.08, 0.30, 0.88, 0.62])
    ax_img.axis("off")
    im = ax_img.imshow(apply_window(hu, init_level, init_width),
                       cmap="gray", vmin=0, vmax=255)

    def make_title(level, width):
        return (f"window level = {level:.0f} HU，width = {width:.0f} HU\n"
                f"顯示範圍 HU [{level - width / 2:.0f}, {level + width / 2:.0f}]")

    ttl = ax_img.set_title(make_title(init_level, init_width))

    # --- 兩條 slider ---
    ax_level = fig.add_axes([0.18, 0.20, 0.68, 0.03])
    ax_width = fig.add_axes([0.18, 0.15, 0.68, 0.03])
    # valfmt 用 "%.0f" 強制走 Python 字串格式化，負數會顯示成 ASCII 的 "-"，
    # 避免某些字型缺 Unicode 減號 U+2212 而跳 glyph 警告。
    s_level = Slider(ax_level, "Level (HU)", LEVEL_MIN, LEVEL_MAX,
                     valinit=init_level, valstep=1, valfmt="%.0f")
    s_width = Slider(ax_width, "Width (HU)", WIDTH_MIN, WIDTH_MAX,
                     valinit=init_width, valstep=1, valfmt="%.0f")

    def redraw(_=None):
        level, width = s_level.val, s_width.val
        im.set_data(apply_window(hu, level, width))
        ttl.set_text(make_title(level, width))
        fig.canvas.draw_idle()

    s_level.on_changed(redraw)
    s_width.on_changed(redraw)

    # --- 預設 window 快捷按鈕 ---
    preset_names = list(WINDOW_PRESETS.keys())
    n = len(preset_names)
    gap, total_w, x0 = 0.02, 0.72, 0.14
    btn_w = (total_w - gap * (n - 1)) / n
    buttons = []  # 保留參考，避免 Button 被 GC 回收後失效

    def make_preset_callback(level, width):
        def _cb(event):
            s_level.set_val(level)   # set_val 會自動觸發 redraw
            s_width.set_val(width)
        return _cb

    for i, name in enumerate(preset_names):
        ax_btn = fig.add_axes([x0 + i * (btn_w + gap), 0.05, btn_w, 0.05])
        b = Button(ax_btn, name)
        level, width = WINDOW_PRESETS[name]
        b.on_clicked(make_preset_callback(level, width))
        buttons.append(b)

    plt.show()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 6 課）：
# 1) 觀察：把 width 拉很窄（例如 100）跟拉很寬（例如 2000），對比和可視
#    範圍分別怎麼變？想想為什麼臨床上不同組織要用不同 window。
#    （提示：width 越窄，對比越強但可視的 HU 範圍越小。）
# 2) 再加一條 slider 控制 gamma，套用 out = 255 * (windowed / 255) ** gamma，
#    看看 gamma 對中間灰階的影響。
# 3) 加一個 Button，把目前 slider 的畫面用 fig.savefig 存成 PNG，
#    檔名帶上當下的 level / width，方便之後對照。
# ------------------------------------------------------------------
