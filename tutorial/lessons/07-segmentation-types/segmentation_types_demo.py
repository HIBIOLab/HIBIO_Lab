"""
第 7 課：語意分割 / 實例分割 / 全景分割
資料：用 numpy 產生一張合成影像，裡面畫了 5 顆大小不一、有些相鄰的『細胞』，
      模擬病理切片細胞計數的情境（沒有現成的小型細胞分割資料集可以免帳號
      下載，若要練真實資料，可參考檔案最下方的 Kaggle 資料集建議）。

這一課不訓練模型，而是動手『產生三種不同的標註方式』，讓你具體看到
semantic / instance / panoptic 三種分割輸出格式的差異。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# 五顆「細胞」的 (中心x, 中心y, 半徑)，其中第 4、5 顆刻意畫得相鄰/重疊
CELLS = [
    (30, 30, 12),
    (80, 40, 15),
    (50, 80, 10),
    (100, 90, 14),
    (118, 100, 10),  # 和上面那顆相鄰，用來凸顯「語意分割分不開它們」
]
IMAGE_SIZE = 140


def make_instance_masks():
    """每顆細胞各自一張 0/1 mask，這就是 instance segmentation 的標註格式。"""
    instance_masks = []
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    for cx, cy, r in CELLS:
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
        instance_masks.append(mask.astype(np.uint8))
    return instance_masks


def to_semantic(instance_masks):
    """語意分割：把所有 instance mask 合併成同一個類別（不分你我）。"""
    semantic = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for mask in instance_masks:
        semantic[mask.astype(bool)] = 1  # 統一標成類別 1 = "cell"
    return semantic


def to_instance_id_map(instance_masks):
    """實例分割：每顆細胞給一個獨立的 instance ID（2, 3, 4 ...），
    後畫的會蓋掉先畫的重疊部分，這也是實例分割要處理『重疊如何仲裁』的地方。"""
    instance_id_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for idx, mask in enumerate(instance_masks, start=1):
        instance_id_map[mask.astype(bool)] = idx
    return instance_id_map


def to_panoptic(instance_masks):
    """全景分割：正式做法是每個像素同時有 (category_id, instance_id) 兩個標籤
    —— stuff（背景組織）像素的 instance_id 沒有意義（設為 0），
    things（細胞）像素則各自有獨立的 instance_id（COCO panoptic 格式的簡化版）。"""
    category_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # 0=stuff, 1=thing(cell)
    instance_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)  # 0=stuff, 1..N=each cell
    for idx, mask in enumerate(instance_masks, start=1):
        m = mask.astype(bool)
        category_map[m] = 1
        instance_map[m] = idx
    return category_map, instance_map


def main():
    instance_masks = make_instance_masks()
    semantic = to_semantic(instance_masks)
    instance_id_map = to_instance_id_map(instance_masks)
    category_map, panoptic_instance_map = to_panoptic(instance_masks)

    n_cells_semantic_can_count = 1  # 語意分割合併成一團，數不出有幾顆
    n_cells_instance_can_count = len(np.unique(instance_id_map)) - 1  # 扣掉背景 0
    print(f"語意分割看到的『類別數』: {n_cells_semantic_can_count}"
          f"（所有細胞都是同一類，無法知道有幾顆）")
    print(f"實例分割看到的『個體數』: {n_cells_instance_can_count}"
          f"（可以正確數出每一顆，包括相鄰的那兩顆）")
    print(f"全景分割: category_map 區分 stuff(0)/thing(1)，"
          f"instance_map 額外標出 {n_cells_instance_can_count} 個獨立細胞 id")

    # 全景分割視覺化：背景(stuff)用淺灰色，每顆細胞(things)用 tab10 個別上色
    cmap = plt.get_cmap("tab10")
    panoptic_rgb = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3)) * 0.85  # 淺灰底色 = stuff
    for idx in range(1, n_cells_instance_can_count + 1):
        panoptic_rgb[panoptic_instance_map == idx] = cmap((idx - 1) % 10)[:3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(semantic, cmap="gray")
    axes[0].set_title("Semantic Segmentation\n(all cells = 1 class)")
    axes[1].imshow(instance_id_map, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("Instance Segmentation\n(each cell = unique ID)")
    axes[2].imshow(panoptic_rgb)
    axes[2].set_title("Panoptic Segmentation\n(gray=stuff, colored=things)")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "07_segmentation_types.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 7 課）：
# 1) 在 CELLS 中再加入一顆跟第 5 顆完全重疊的細胞，觀察 instance_id_map
#    中「後畫的蓋掉先畫的」這個簡化假設會造成什麼問題？實務上（如
#    Mask R-CNN）是怎麼處理重疊物件的？
# 2) 想用真實資料練習實例分割，可以嘗試 Kaggle 的
#    "2018 Data Science Bowl"（細胞核實例分割）資料集：
#    https://www.kaggle.com/competitions/data-science-bowl-2018
#    下載後，把每張圖對應的多個 mask 檔案，比照本程式的
#    make_instance_masks() 合併方式重寫即可（需要自行以 Kaggle 帳號登入
#    下載資料）。
# ------------------------------------------------------------------
