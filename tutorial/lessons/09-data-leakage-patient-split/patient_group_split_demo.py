"""
第 9 課：Data Leakage 與病人層級切分 / Data Augmentation
資料集：Breast Cancer Wisconsin (Diagnostic)（sklearn 內建）

Breast Cancer 資料集本身「一列 = 一位病人」，沒有天生的分組結構，所以這裡
用「複製每一位病人的資料列 3 次、加一點點雜訊」來模擬『同一位病人有多張
切片 / 多次追蹤影像』的情境 —— 這是為了教學示範才刻意製造的分組，
真實醫學影像資料集通常本來就會有 patient_id 這個欄位。

比較兩種切分方式：
  (A) 錯誤做法：忽略病人分組，直接對所有『切片』隨機切 train/test
      -> 同一位病人的切片同時出現在兩邊，造成 leakage，分數虛高。
  (B) 正確做法：用 GroupKFold，以病人為單位切分。
"""

import sys

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold, cross_val_score


def make_simulated_patient_slices(n_slices_per_patient=3, noise_std=1e-5, seed=42):
    """把每位『病人』複製成多張『切片』，並加入極小的雜訊模擬同一顆腫瘤在
    相鄰切片上幾乎一樣的情況——這正是 leakage 最容易發生的場景：切片幾乎是
    彼此的近似複製品，模型可以直接『認出』而不是真正學到病灶特徵。"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    rng = np.random.default_rng(seed)

    X_list, y_list, patient_ids = [], [], []
    for patient_id, (x_row, y_row) in enumerate(zip(X, y)):
        for _ in range(n_slices_per_patient):
            noise = rng.normal(0, noise_std, size=x_row.shape) * x_row.std()
            X_list.append(x_row + noise)
            y_list.append(y_row)
            patient_ids.append(patient_id)

    return np.array(X_list), np.array(y_list), np.array(patient_ids)


def compare(name, model, X, y, patient_ids):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    leaky = cross_val_score(model, X, y, cv=kfold, scoring="accuracy").mean()

    group_kfold = GroupKFold(n_splits=5)
    honest = cross_val_score(model, X, y, cv=group_kfold, groups=patient_ids, scoring="accuracy").mean()

    print(f"{name:38s} 錯誤切分(KFold)={leaky:.4f}   "
          f"正確切分(GroupKFold)={honest:.4f}   差距={leaky - honest:+.4f}")
    return leaky, honest


def main():
    X, y, patient_ids = make_simulated_patient_slices()
    print(f"模擬資料：{len(np.unique(patient_ids))} 位病人 x 3 張切片 = {len(y)} 筆"
          f"（切片之間只加了極小的雜訊，模擬近乎重複的相鄰切片）\n")

    print("== 錯誤切分 vs 正確切分，比較不同模型的『記憶力』差異 ==")
    knn1 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    compare("KNN (k=1，最容易背答案)", knn1, X, y, patient_ids)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    compare("Random Forest", rf, X, y, patient_ids)

    print("\n=> KNN(k=1) 在錯誤切分下幾乎能得到接近滿分的準確率，因為 train 裡")
    print("   幾乎必定存在同一位病人的『近乎複製』切片，直接抓最近的鄰居就能")
    print("   蒙對答案；GroupKFold 排除這個漏洞後，準確率明顯掉下來，這才是")
    print("   模型面對『真正沒看過的病人』時的實力。RandomForest 比較不會靠")
    print("   單一最近鄰作弊，所以差距通常比 KNN 小，但一樣存在。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 9 課）：
# 1) 把 noise_std 調大（例如 0.1 甚至 0.3），讓同一位病人的切片彼此差異
#    變大，重新比較 KNN 的『錯誤切分 vs 正確切分』差距是否縮小？這說明了
#    『病人內變異 vs 病人間變異』如何影響 leakage 的嚴重程度。
# 2) 找出下面 pipeline 的錯誤並修正：
#      "先對全部影像資料做 StandardScaler().fit_transform(X)，
#       再用 train_test_split 隨機切分成 train/test。"
#    （提示：StandardScaler 應該只 fit 在哪一份資料上？）
# 3) 用 torchvision.transforms 寫一個包含 RandomRotation、
#    RandomHorizontalFlip、ColorJitter 的 augmentation pipeline，
#    套用在幾張 MNIST 圖片上並存成圖片比較前後差異；再想想：如果這是
#    一張『胸部 X 光』，RandomHorizontalFlip 還適合直接套用嗎？
# ------------------------------------------------------------------
