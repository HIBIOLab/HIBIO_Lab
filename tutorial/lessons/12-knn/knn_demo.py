"""
第 12 課：K-Nearest Neighbors (KNN)
資料集：COVID-19 Cases Prediction（Delphi group / 李宏毅 ML2022 HW01）
        https://github.com/virginiakm1988/ML2022-Spring/tree/main/HW01

回歸任務：用 116 個特徵（37 州 one-hot ＋ 連續 5 天的問卷/症狀指標）預測某一州
某一天的 COVID 陽性率。這裡用 KNeighborsRegressor（找最近的 K 個鄰居，取平均），
示範三件事：
  1) 比較「有做 feature scaling」vs「沒做」對 KNN 的影響 —— COVID 特徵尺度差異
     很大（州是 0/1，問卷指標是 0~100），但這裡會看到一個反例：當多數特徵其實
     是冗餘的，naive 地把全部特徵 scaling 到同等權重反而可能稀釋有用訊號，帶出
     「scaling 要搭配特徵篩選」的觀念。
  2) 掃描不同的 K 值，觀察 bias-variance trade-off。
  3) 一個小型合成實驗，展示『維度詛咒』：維度越高，最近與最遠的距離差異越小，
     距離這個概念本身越沒有鑑別力。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "covid"
TRAIN_CSV = DATA_DIR / "covid.train.csv"
GITHUB = "https://github.com/virginiakm1988/ML2022-Spring/tree/main/HW01"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_covid():
    if not TRAIN_CSV.exists():
        return None
    with open(TRAIN_CSV, newline="") as f:
        rows = list(csv.reader(f))
    data = np.array(rows[1:], dtype=np.float64)
    return data[:, 1:-1], data[:, -1]   # X（去掉 id 與 target）, y（第 5 天陽性率）


def print_missing_data_help():
    print(f"找不到資料集: {TRAIN_CSV}")
    print("這一章用的是 COVID-19 Cases Prediction 資料集（Delphi group / ML2022 HW01）。")
    print("請到下面的網址下載 covid.train.csv、covid.test.csv，放到 tutorial/data/covid/：")
    print(f"  {GITHUB}")


def scaling_matters(X, y):
    knn = KNeighborsRegressor(n_neighbors=5)
    r2_raw = cross_val_score(knn, X, y, cv=5, scoring="r2").mean()
    X_scaled = StandardScaler().fit_transform(X)
    r2_scaled = cross_val_score(knn, X_scaled, y, cv=5, scoring="r2").mean()

    print("== Feature Scaling 對 KNN 的影響 (5-fold CV R2，越接近 1 越好) ==")
    print(f"沒有 scaling: R2 = {r2_raw:.4f}")
    print(f"有 scaling  : R2 = {r2_scaled:.4f}")
    print("=> KNN 用『距離』找鄰居，特徵的尺度會直接決定誰主導距離。教科書常說")
    print("   『KNN 一定要先 scaling』——那是因為當各特徵尺度差很大、又同樣重要時，")
    print("   不 scaling 會讓大尺度特徵獨佔距離。但 COVID 這份資料是個有趣的反例：")
    print("   116 個特徵裡真正有預測力的其實只有少數『最近幾天的陽性率』(見第 14、")
    print("   15 課)，它們本來就是 0~100 的大尺度，不 scaling 時距離剛好由這些有用")
    print("   特徵主導，分數反而略高；一旦 scaling 把 37 個 one-hot 州別和一堆冗餘")
    print("   欄位全拉到同等權重，有用的訊號反而被稀釋。")
    print("   結論：scaling 讓特徵站上同一個尺度，但『同尺度』只有在特徵同樣重要時")
    print("   才是好事——所以實務上常要 scaling 搭配特徵篩選一起用。\n")
    return X_scaled


def k_sweep(X_scaled, y):
    print("== 不同 K 值的 5-fold CV R2（bias-variance trade-off） ==")
    ks = list(range(1, 31, 2))
    r2s = []
    for k in ks:
        knn = KNeighborsRegressor(n_neighbors=k)
        r2 = cross_val_score(knn, X_scaled, y, cv=5, scoring="r2").mean()
        r2s.append(r2)
        print(f"  K={k:2d}: R2={r2:.4f}")

    best_k = ks[int(np.argmax(r2s))]
    print(f"\n最佳 K = {best_k}（R2={max(r2s):.4f}）")
    print("=> K 太小（如 K=1）容易被個別雜訊樣本影響（overfit / high variance）；")
    print("   K 太大則會把太多不相關的樣本也算進來平均，預測過度平滑（underfit）。")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, r2s, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"最佳 K={best_k}")
    ax.set_xlabel("K")
    ax.set_ylabel("5-fold CV R2")
    ax.set_title("KNN Regressor: R2 vs K（COVID 陽性率預測）")
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
    print("=> 維度越高，比值越接近 1，代表『最近的鄰居』跟『最遠的鄰居』幾乎一樣")
    print("   遠 —— 距離失去鑑別力。COVID 有 116 個特徵，這也是為什麼先用第 14、15")
    print("   課的方法篩掉冗餘特徵，往往能讓 KNN 這類靠距離的模型表現更好。")


def main():
    loaded = load_covid()
    if loaded is None:
        print_missing_data_help()
        return
    X, y = loaded
    print(f"資料集: {X.shape[0]} 筆 (state-day)，{X.shape[1]} 個特徵，回歸目標 = 第 5 天陽性率\n")
    X_scaled = scaling_matters(X, y)
    k_sweep(X_scaled, y)
    curse_of_dimensionality()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 12 課）：
# 1) 一般都說 KNN 前要先 feature scaling，但這裡 scaling 後 R2 反而略降。結合
#    scaling_matters() 的結果說明原因：scaling 什麼時候幫得上忙、什麼時候反而
#    稀釋了少數真正有用的特徵？（提示：跟「多數特徵是冗餘的」有關）
# 2) 在 116 個特徵上直接用 KNN 會遇到維度詛咒。可以怎麼緩解（提示：先用
#    第 14 課 Random Forest 的 feature importance，或第 15 課的 Lasso 做特徵
#    篩選，只留最重要的幾個特徵再跑 KNN）？動手比較篩選前後的 R2。
# ------------------------------------------------------------------
