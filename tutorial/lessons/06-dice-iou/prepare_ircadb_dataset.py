"""
第 6 課 - 資料前處理：把 3D-IRCADb-01 原始 DICOM 轉成訓練用的 2D 切片
資料來源：https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/
（需要免費註冊帳號才能下載，我們不會替你處理帳密或下載檔案）

⚠️ 重要：3D-IRCADb-01 是「多層壓縮」的！外層 3Dircadb1.zip 解開後，每一位病人是
一個資料夾，但資料夾裡的 PATIENT_DICOM、MASKS_DICOM 等其實還是各自的 .zip：

    3Dircadb1.1/
        PATIENT_DICOM.zip       <- 解開後才是 CT 原始切片（DICOM）
        MASKS_DICOM.zip         <- 解開後才是各器官遮罩（含 liver/、livertumor01/…）
        LABELLED_DICOM.zip      <- 這一課用不到
        MESHES_VTK.zip          <- 這一課用不到
    3Dircadb1.2/
        ...
    ...最多到 3Dircadb1.20/

這支程式會「自動幫你解開這些內層 zip」，不用手動一個一個解。解開後 PATIENT_DICOM/
是 CT 切片，MASKS_DICOM/liver/ 是肝臟遮罩、MASKS_DICOM/livertumor01/（02, 03…）是
腫瘤遮罩。接著把它們轉換成訓練程式（train_attention_unet_ircadb.py）容易讀取的
格式：每個病人每張切片存成一個 .npz 檔，包含 CT 影像、肝臟遮罩、腫瘤遮罩（多顆
腫瘤會自動合併成一張遮罩），並套用第 8 課教過的 HU windowing。

用法：
    1) 到上面的網址下載並解壓縮「外層」資料集，放到 tutorial/data/3Dircadb1/，
       裡面應該要能看到 3Dircadb1.1/、3Dircadb1.2/ ... 這些資料夾（內層的
       PATIENT_DICOM.zip 等維持壓縮狀態沒關係，這支程式會自動解開）。
    2) python prepare_ircadb_dataset.py
    3) 轉換好的切片會存到 tutorial/data/ircadb_slices/，train_attention_unet_ircadb.py
       會直接讀這個資料夾。
"""

import sys
import zipfile
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pydicom

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "3Dircadb1"          # 使用者下載解壓縮後放這裡
OUT_DIR = DATA_DIR / "ircadb_slices"       # 轉換後的輸出

IMAGE_SIZE = 256          # 跟參考 notebook 一致，統一 resize 成 256x256
WINDOW_LEVEL, WINDOW_WIDTH = 40, 400  # 第 8 課教過的『軟組織窗』，適合看肝臟


def ensure_dir(parent: Path, name: str) -> Path:
    """確保 parent/name 是一個資料夾。如果它目前只是一個 parent/name.zip，
    就自動解壓縮成資料夾再回傳。這樣不管 IRCAD 把資料打包成幾層 zip 都能處理。

    解壓縮時會判斷 zip 內部結構：
      - 若 zip 內已經包了一層同名資料夾（例如 PATIENT_DICOM/image_0），
        就解到 parent，結果是 parent/PATIENT_DICOM/image_0。
      - 若 zip 內是散檔或其他結構，就解到 parent/name/ 底下，
        結果一樣是 parent/name/... 。
    """
    target = parent / name
    if target.is_dir():
        return target
    zpath = parent / f"{name}.zip"
    if zpath.is_file():
        print(f"    解壓縮 {zpath.parent.name}/{zpath.name} ...")
        try:
            with zipfile.ZipFile(zpath) as zf:
                entries = [n for n in zf.namelist() if not n.endswith("/")]
                tops = {n.replace("\\", "/").split("/")[0] for n in entries}
                if tops == {name}:
                    zf.extractall(parent)
                else:
                    target.mkdir(parents=True, exist_ok=True)
                    zf.extractall(target)
        except zipfile.BadZipFile:
            print(f"    [警告] {zpath.name} 不是有效的 zip，跳過")
    return target


def read_dicom_series(folder: Path):
    """讀取一個資料夾（含子資料夾）內所有 DICOM 檔案，回傳 {檔名: dataset}。
    IRCAD 的檔名（例如 image_0, image_1）沒有 .dcm 副檔名，也可能沒有標準
    DICOM 檔頭，所以用 force=True 讀取；用 rglob 遞迴是為了容忍解壓縮後可能
    多出來的一層巢狀資料夾。"""
    slices = {}
    if not folder.exists():
        return slices
    for f in sorted(folder.rglob("*")):
        if not f.is_file() or f.suffix.lower() == ".zip":
            continue
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            slices[f.name] = ds
        except Exception:
            continue
    return slices


def apply_window(hu_array, level=WINDOW_LEVEL, width=WINDOW_WIDTH):
    """跟第 8 課一樣的 windowing，把 HU 值壓成 0-1 的浮點數。"""
    low, high = level - width / 2, level + width / 2
    windowed = np.clip(hu_array, low, high)
    return (windowed - low) / (high - low)


def resize_nearest(arr, size):
    """簡單的最近鄰縮放，避免額外依賴 scikit-image / opencv。"""
    h, w = arr.shape
    if (h, w) == (size, size):
        return arr
    row_idx = (np.arange(size) * h / size).astype(int).clip(0, h - 1)
    col_idx = (np.arange(size) * w / size).astype(int).clip(0, w - 1)
    return arr[row_idx][:, col_idx]


def find_tumor_dirs(masks_dicom_dir: Path):
    """找出所有 livertumor01, livertumor02... 資料夾（有些病人沒有腫瘤）。
    若腫瘤遮罩本身也是 zip（livertumor01.zip…）會先自動解開。"""
    if not masks_dicom_dir.is_dir():
        return []
    # 先把任何 tumor 相關的 zip 解開
    for z in sorted(masks_dicom_dir.glob("*.zip")):
        if "tumor" in z.stem.lower():
            ensure_dir(masks_dicom_dir, z.stem)
    return sorted(d for d in masks_dicom_dir.iterdir()
                  if d.is_dir() and "tumor" in d.name.lower())


def process_patient(patient_dir: Path, out_dir: Path):
    patient_id = patient_dir.name  # 例如 "3Dircadb1.1"

    # 自動解開內層 zip（若已是資料夾則直接使用）
    patient_dicom = ensure_dir(patient_dir, "PATIENT_DICOM")
    masks_dicom = ensure_dir(patient_dir, "MASKS_DICOM")

    patient_scans = read_dicom_series(patient_dicom)
    liver_masks = read_dicom_series(ensure_dir(masks_dicom, "liver"))
    tumor_dirs = find_tumor_dirs(masks_dicom)

    if not patient_scans:
        print(f"  [跳過] {patient_id}：找不到 PATIENT_DICOM 的切片")
        return 0

    n_saved = 0
    for fname, ds in patient_scans.items():
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        hu = ds.pixel_array.astype(np.float32) * slope + intercept
        img = apply_window(hu)
        img = resize_nearest(img, IMAGE_SIZE).astype(np.float32)

        if fname in liver_masks:
            liver = (liver_masks[fname].pixel_array > 0).astype(np.uint8)
            liver = resize_nearest(liver, IMAGE_SIZE)
        else:
            liver = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

        tumor = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for tdir in tumor_dirs:
            tumor_masks = read_dicom_series(tdir)
            if fname in tumor_masks:
                t = (tumor_masks[fname].pixel_array > 0).astype(np.uint8)
                tumor |= resize_nearest(t, IMAGE_SIZE)

        out_path = out_dir / f"{patient_id}_{fname}.npz"
        np.savez_compressed(out_path, image=img, liver=liver, tumor=tumor)
        n_saved += 1

    print(f"  {patient_id}: 轉換 {n_saved} 張切片")
    return n_saved


def main():
    if not RAW_DIR.exists():
        print(f"找不到原始資料夾: {RAW_DIR}")
        print("請先到官網下載 3D-IRCADb-01 並解壓縮「外層」到這個路徑：")
        print("  https://www.ircad.fr/research-and-development/data-sets/liver-segmentation-3d-ircadb-01/")
        print(f"解壓縮後的路徑應該長這樣：{RAW_DIR}/3Dircadb1.1/, {RAW_DIR}/3Dircadb1.2/, ...")
        print("（裡面的 PATIENT_DICOM.zip 等內層 zip 不用手動解，這支程式會自動處理）")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 有些下載版本連「病人資料夾」本身也是 zip（例如 3Dircadb1.1.zip），先自動解開
    for z in sorted(RAW_DIR.glob("*.zip")):
        ensure_dir(RAW_DIR, z.stem)

    patient_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    if not patient_dirs:
        print(f"在 {RAW_DIR} 裡找不到任何病人資料夾（應該要有 3Dircadb1.1/ 等）。")
        return
    print(f"找到 {len(patient_dirs)} 位病人的資料夾，開始轉換"
          f"（會自動解開內層 PATIENT_DICOM.zip / MASKS_DICOM.zip）...")

    total = 0
    for pdir in patient_dirs:
        total += process_patient(pdir, OUT_DIR)

    print(f"\n完成！總共轉換 {total} 張切片，存到 {OUT_DIR}")


if __name__ == "__main__":
    main()
