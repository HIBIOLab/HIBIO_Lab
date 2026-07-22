"""
第 13 課：K-Means Clustering
資料集：Digits（sklearn 內建，1797 張 8x8 手寫數字小圖，10 個類別）

K-means 是非監督式學習，這裡刻意「假裝不知道標籤」，只用像素特徵去分群，
再拿真實標籤回頭檢查分群結果、順便展示如何用 elbow method / silhouette
score 決定 K。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    data = load_digits()
    X, y_true = data.data, data.target
    X_scaled = StandardScaler().fit_transform(X)
    print(f"資料筆數: {len(y_true)}，特徵維度: {X.shape[1]}（8x8 像素攤平）\n")

    print("== 用 elbow method（inertia）與 silhouette score 找合適的 K ==")
    ks = range(2, 16)
    inertias, silhouettes = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  K={k:2d}: inertia={km.inertia_:9.1f}   silhouette={silhouettes[-1]:.4f}")

    best_k_by_silhouette = list(ks)[int(np.argmax(silhouettes))]
    print(f"\nSilhouette score 建議的 K = {best_k_by_silhouette}")
    print("（我們知道真實類別數是 10，這裡可以觀察 silhouette 選出的 K 是否接近 10；")
    print(" 不完全相等很正常，因為有些數字寫法很像，K-means 可能會混在一起）\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(list(ks), inertias, marker="o")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(list(ks), silhouettes, marker="o", color="orange")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score vs K")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "13_kmeans_elbow_silhouette.png", dpi=120)

    print("== 用 K=10 跑 K-means，看看每個 cluster 的『中心影像』長什麼樣 ==")
    km10 = KMeans(n_clusters=10, n_init=10, random_state=42)
    labels10 = km10.fit_predict(X_scaled)

    # 拿真實標籤檢查：每個 cluster 裡最常見的真實數字是哪個
    for cluster_id in range(10):
        mask = labels10 == cluster_id
        if mask.sum() == 0:
            continue
        most_common_digit = np.bincount(y_true[mask]).argmax()
        purity = (y_true[mask] == most_common_digit).mean()
        print(f"  Cluster {cluster_id}: {mask.sum():4d} 筆，最多的真實數字是 "
              f"'{most_common_digit}'（純度 {purity:.1%}）")

    fig2, axes2 = plt.subplots(2, 5, figsize=(10, 4.5))
    centers_img = km10.cluster_centers_.reshape(10, 8, 8)
    for i, ax in enumerate(axes2.ravel()):
        ax.imshow(centers_img[i], cmap="gray")
        ax.set_title(f"cluster {i}")
        ax.axis("off")
    fig2.suptitle("K-Means Cluster Centers（還原成 8x8 影像）")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "13_kmeans_cluster_centers.png", dpi=120)
    print(f"\n圖片已存到: {OUTPUT_DIR / '13_kmeans_elbow_silhouette.png'}")
    print(f"圖片已存到: {OUTPUT_DIR / '13_kmeans_cluster_centers.png'}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 13 課）：
# 1) 簡述 K-means 演算法的兩個交替步驟（assignment / update），可以在
#    程式中找到 sklearn KMeans 對應的行為並用註解標出來。
# 2) 觀察上面印出的『每個 cluster 純度』，哪幾個 cluster 純度特別低？
#    猜猜看是哪些數字容易被 K-means 搞混（提示：形狀相似的數字，如
#    4/9、3/8、1/7），這對應到 index.html 中提到的『K-means 假設球型
#    分佈、對形狀複雜的資料效果有限』。
# ------------------------------------------------------------------
