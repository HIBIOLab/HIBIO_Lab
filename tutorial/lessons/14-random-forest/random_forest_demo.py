"""
第 14 課：Random Forest
資料集：COVID-19 Cases Prediction（Delphi group / 李宏毅 ML2022 HW01）
        https://github.com/virginiakm1988/ML2022-Spring/tree/main/HW01

回歸任務：用 116 個特徵（37 州 one-hot ＋ 連續 5 天的問卷/症狀指標）預測某一州
某一天的 COVID 陽性率。這裡用 RandomForestRegressor，和單棵 Decision Tree、
一般 Linear Regression 比較 5-fold CV 的表現（穩定性 vs 單棵樹），再印出
feature importance —— 你會清楚看到「最近幾天的陽性率」主導了預測。
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "covid"
TRAIN_CSV = DATA_DIR / "covid.train.csv"
GITHUB = "https://github.com/virginiakm1988/ML2022-Spring/tree/main/HW01"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DAY_FEATURES = [
    "cli", "ili", "hh_cmnty_cli", "nohh_cmnty_cli", "wearing_mask",
    "travel_outside_state", "work_outside_home", "shop", "restaurant",
    "spent_time", "large_event", "public_transit", "anxious", "depressed",
    "worried_finances", "tested_positive",
]


def make_feature_names(header):
    """州名維持原樣，5 天的問卷指標加上 d1_ ~ d5_ 前綴。"""
    feats = header[1:-1]
    start = feats.index("cli")
    named = list(feats[:start])
    day, k = 1, 0
    for name in feats[start:]:
        named.append(f"d{day}_{name}")
        k += 1
        if k % len(DAY_FEATURES) == 0:
            day += 1
    return named


def load_covid():
    if not TRAIN_CSV.exists():
        return None
    with open(TRAIN_CSV, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    data = np.array(rows[1:], dtype=np.float64)
    return data[:, 1:-1], data[:, -1], make_feature_names(header)


def print_missing_data_help():
    print(f"找不到資料集: {TRAIN_CSV}")
    print("這一章用的是 COVID-19 Cases Prediction 資料集（Delphi group / ML2022 HW01）。")
    print("請到下面的網址下載 covid.train.csv、covid.test.csv，放到 tutorial/data/covid/：")
    print(f"  {GITHUB}")


def main():
    loaded = load_covid()
    if loaded is None:
        print_missing_data_help()
        return
    X, y, feature_names = loaded
    print(f"資料集: {X.shape[0]} 筆 (state-day)，{X.shape[1]} 個特徵，回歸目標 = 第 5 天陽性率\n")

    print("== 三種模型的 5-fold CV 表現比較（用 R2，越接近 1 越好）==")
    models = {
        "Decision Tree (單棵樹)": DecisionTreeRegressor(random_state=42),
        "Random Forest (200 棵樹)": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Linear Regression": LinearRegression(),
    }
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        print(f"  {name:28s} R2 mean={scores.mean():.4f}  std={scores.std():.4f}")
    print("=> 單棵 Decision Tree 很容易對訓練資料的雜訊過擬合，R2 通常最不穩；")
    print("   Random Forest 用 bagging（多棵樹在不同 bootstrap 子集上訓練、再平均）")
    print("   有效降低這種 variance，在沒看過的資料上更穩定。\n")

    print("== Random Forest 的 Feature Importance（哪些欄位真的在預測陽性率）==")
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:10]:
        print(f"  {feature_names[i]:28s} importance={importances[i]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    top_n = 10
    ax.barh([feature_names[i] for i in order[:top_n]][::-1], importances[order[:top_n]][::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest: Top 10 重要特徵（COVID 陽性率預測）")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "14_random_forest_importance.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")
    print("=> feature importance 幾乎被『最近幾天的 tested_positive』佔滿，非常符合")
    print("   直覺：預測明天的陽性率，最有用的線索就是最近幾天的陽性率。這種排序")
    print("   也常被拿來做特徵篩選——只留最重要的幾個特徵再建模，兼顧可解釋性與降維。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 14 課）：
# 1) 解釋 bagging（自助抽樣聚合）如何幫助降低模型 variance？可以對照上面
#    Decision Tree 和 Random Forest 的 R2 平均與標準差來說明；也可以試著把
#    DecisionTreeRegressor(max_depth=5) 加上深度限制，觀察是否更接近 RF。
# 2) 只取 feature importance 排名前 5 的特徵重新訓練 Random Forest，比較 R2
#    跟用全部 116 個特徵時差多少？這說明了什麼？
# ------------------------------------------------------------------
