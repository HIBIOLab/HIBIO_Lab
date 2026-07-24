"""
第 15 課：Ridge / Lasso Regression（正則化）
資料集：COVID-19 Cases Prediction（Delphi group / 李宏毅 ML2022 HW01）
        https://github.com/virginiakm1988/ML2022-Spring/tree/main/HW01

這是一個回歸任務：用 116 個特徵（37 州 one-hot ＋ 連續 5 天的問卷/症狀指標）
預測某一州某一天的 COVID 陽性率（tested_positive，連續值）。116 個特徵裡有大量
彼此高度相關的欄位（連續 5 天的同一種指標幾乎一路連動），一般線性迴歸在這種
共線性下係數容易亂飄、不穩定；這一課就用它來比較 Linear / Ridge (L2) / Lasso (L1)，
親眼看到 Lasso 如何把一堆冗餘特徵的係數直接壓成 0，篩出一小組真正有預測力的特徵。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import csv
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "covid"
TRAIN_CSV = DATA_DIR / "covid.train.csv"
GITHUB = "https://github.com/virginiakm1988/ML2022-Spring/tree/main/HW01"

DAY_FEATURES = [
    "cli", "ili", "hh_cmnty_cli", "nohh_cmnty_cli", "wearing_mask",
    "travel_outside_state", "work_outside_home", "shop", "restaurant",
    "spent_time", "large_event", "public_transit", "anxious", "depressed",
    "worried_finances", "tested_positive",
]


def make_feature_names(header):
    """把 CSV 標頭轉成可讀、不重複的特徵名：州名維持原樣，5 天的問卷指標
    加上 d1_ ~ d5_ 前綴（例如 d4_tested_positive = 第 4 天的陽性率）。"""
    feats = header[1:-1]            # 去掉 id 與最後一欄 target
    start = feats.index("cli")      # 州 one-hot 之後就是問卷指標
    named = list(feats[:start])     # 州名照舊
    day, k = 1, 0
    for name in feats[start:]:
        named.append(f"d{day}_{name}")
        k += 1
        if k % len(DAY_FEATURES) == 0:
            day += 1
    return named


def load_covid():
    """回傳 (X, y, feature_names)；找不到資料集則回傳 None。"""
    if not TRAIN_CSV.exists():
        return None
    with open(TRAIN_CSV, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array(rows[1:], dtype=np.float64)
    X = data[:, 1:-1]   # 去掉 id 與 target
    y = data[:, -1]     # 第 5 天 tested_positive（回歸目標）
    return X, y, make_feature_names(header)


def print_missing_data_help():
    print(f"找不到資料集: {TRAIN_CSV}")
    print("這一章用的是 COVID-19 Cases Prediction 資料集（Delphi group / ML2022 HW01）。")
    print("請到下面的網址下載 covid.train.csv、covid.test.csv，放到 tutorial/data/covid/：")
    print(f"  {GITHUB}")


def evaluate(model, Xtr, Xva, ytr, yva):
    model.fit(Xtr, ytr)
    rmse = mean_squared_error(yva, model.predict(Xva)) ** 0.5
    n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-6))
    return {
        "train_r2": r2_score(ytr, model.predict(Xtr)),
        "val_r2": r2_score(yva, model.predict(Xva)),
        "val_rmse": rmse,
        "n_nonzero": n_nonzero,
    }


def main():
    loaded = load_covid()
    if loaded is None:
        print_missing_data_help()
        return
    X, y, feat_names = loaded
    print(f"資料集: {X.shape[0]} 筆 (state-day)，{X.shape[1]} 個特徵，回歸目標 = 第 5 天陽性率")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge (L2, alpha=10)": Ridge(alpha=10.0),
        "Lasso (L1, alpha=0.1)": Lasso(alpha=0.1, max_iter=10000),
    }

    print(f"\n{'模型':22s} {'train R2':>9s} {'val R2':>8s} {'val RMSE':>9s} {'非零係數':>10s}")
    print("-" * 64)
    lasso = None
    for name, model in models.items():
        m = evaluate(model, Xtr, Xva, ytr, yva)
        print(f"{name:22s} {m['train_r2']:9.4f} {m['val_r2']:8.4f} "
              f"{m['val_rmse']:9.4f} {m['n_nonzero']:6d}/{X.shape[1]}")
        if name.startswith("Lasso"):
            lasso = model

    # Lasso 篩出來的特徵（依係數絕對值排序）
    order = np.argsort(np.abs(lasso.coef_))[::-1]
    kept = [i for i in order if abs(lasso.coef_[i]) > 1e-6]
    print(f"\nLasso 只保留 {len(kept)} 個特徵，影響力最大的前 10 個：")
    for i in kept[:10]:
        print(f"  {feat_names[i]:26s} coef = {lasso.coef_[i]:+.4f}")

    print("\n=> Linear / Ridge 幾乎用滿全部 116 個特徵，Lasso 卻只靠一小組特徵就達到")
    print("   相近的 val R2——而且被留下來的通常是『最近幾天的陽性率』這種真的有")
    print("   預測力的欄位。這就是 L1 正則化『自動特徵篩選』的價值：模型更精簡、")
    print("   也更好解讀。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 15 課）：
# 1) 對照印出的「非零係數個數」，說明 L1 (Lasso) 與 L2 (Ridge) 在「特徵篩選」
#    這一點的關鍵差異：為什麼 Ridge 幾乎不會把係數壓到剛好 0？
# 2) 調整 Lasso 的 alpha（例如 0.01、0.1、1.0），觀察非零係數個數與 val R2
#    如何隨 alpha 變化，alpha 太大或太小分別會發生什麼問題？
# 3) 只用 Lasso 篩出來的那幾個特徵重新訓練一個一般 Linear Regression，val R2
#    掉了多少？這說明 116 個特徵裡有多少其實是冗餘的？
# ------------------------------------------------------------------
