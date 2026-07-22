"""
第 12 課：K-Nearest Neighbors (KNN)
資料集：Breast Cancer Wisconsin (Diagnostic)（sklearn 內建）

三件事：
  1) 比較「有做 feature scaling」vs「沒做」對 KNN 準確率的影響。
  2) 掃描不同的 K 值，觀察 bias-variance trade-off。
  3) 一個小型合成實驗，展示『維度詛咒』：維度越高，最近與最遠的距離
     差異越小，距離這個概念本身越沒有鑑別力。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def scaling_matters():
    data = load_breast_cancer()
    X, y = data.data, data.target

    knn = KNeighborsClassifier(n_neighbors=5)
    acc_raw = cross_val_score(knn, X, y, cv=5, scoring="accuracy").mean()

    X_scaled = StandardScaler().fit_transform(X)
    acc_scaled = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy").mean()

    print("== Feature Scaling 對 KNN 的影響 (5-fold CV accuracy) ==")
    print(f"沒有 scaling: {acc_raw:.4f}")
    print(f"有 scaling  : {acc_scaled:.4f}")
    print("=> KNN 是用『距離』來判斷鄰居，如果某個特徵的數值範圍特別大")
    print("   （例如 area 動輒上千，但 smoothness 只有 0 點多），沒有 scaling")
    print("   的話距離幾乎完全被那個大範圍的特徵主宰，其他特徵形同虛設。\n")
    return X_scaled, y


def k_sweep(X_scaled, y):
    print("== 不同 K 值的 5-fold CV accuracy（bias-variance trade-off） ==")
    ks = list(range(1, 31, 2))
    accs = []
    for k in ks:
        knn = KNeighborsClassifier(n_neighbors=k)
        acc = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy").mean()
        accs.append(acc)
        print(f"  K={k:2d}: accuracy={acc:.4f}")

    best_k = ks[int(np.argmax(accs))]
    print(f"\n最佳 K = {best_k}（accuracy={max(accs):.4f}）")
    print("=> K 太小（如 K=1）容易被雜訊樣本影響（overfit / high variance）；")
    print("   K 太大則會把太多不相關的樣本也算進來，決策邊界過度平滑（underfit）。")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, accs, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"最佳 K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_title("KNN: Accuracy vs K")
    ax.legend()
    fig.tight_layout()
    out_path = OUTPUT_DIR / "12_knn_k_sweep.png"
    fig.savefig(out_path, dpi=120)
    print(f"圖片已存到: {out_path}\n")


def curse_of_dimensionality():
    print("== 維度詛咒示範：隨機點在不同維度下，最近/最遠距離的比值 ==")
    rng = np.random.default_rng(0)
    n_points = 1000
    for dim in [2, 10, 50, 200, 1000]:
        pts = rng.uniform(0, 1, size=(n_points, dim))
        query = rng.uniform(0, 1, size=(1, dim))
        dists = np.linalg.norm(pts - query, axis=1)
        ratio = dists.min() / dists.max()
        print(f"  維度={dim:5d}:  最近/最遠距離比值 = {ratio:.4f}")
    print("=> 維度越高，比值越接近 1，代表『最近的鄰居』跟『最遠的鄰居』")
    print("   幾乎一樣遠 —— 距離失去鑑別力，這就是 KNN 在高維度（例如上百個")
    print("   radiomics 特徵）表現變差的原因之一。")


if __name__ == "__main__":
    X_scaled, y = scaling_matters()
    k_sweep(X_scaled, y)
    curse_of_dimensionality()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 12 課）：
# 1) 為什麼使用 KNN 前必須先做 feature scaling？結合 scaling_matters()
#    的實驗結果說明。
# 2) 在高維度（例如 radiomics 上百個特徵）資料上直接用 KNN，可能遇到
#    什麼問題？可以怎麼緩解（提示：先用第 14 課 Random Forest 的
#    feature importance，或第 15 課的 Lasso 做特徵篩選再用 KNN）？
# ------------------------------------------------------------------
