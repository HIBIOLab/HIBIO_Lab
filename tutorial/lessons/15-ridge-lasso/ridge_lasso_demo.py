"""
第 15 課：Ridge / Lasso Regression（正則化）
資料集：sklearn.datasets.make_regression 產生的合成資料，刻意模擬
        「radiomics 特徵數遠大於病人數」的情境 —— 300 個特徵、只有 80 位
        『病人』，其中只有 15 個特徵是真正有預測力的，其餘都是雜訊。

比較一般 Linear Regression、Ridge (L2)、Lasso (L1) 在這種小樣本高維度
資料上的表現，具體看到 Lasso 如何把無用特徵的係數直接壓成 0。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

N_SAMPLES = 80       # 模擬 80 位病人
N_FEATURES = 300     # 模擬 300 個 radiomics 特徵
N_INFORMATIVE = 15   # 其中只有 15 個特徵真正有用，其餘都是雜訊


def main():
    X, y, true_coef = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=N_INFORMATIVE,
        noise=10.0,
        coef=True,
        random_state=42,
    )
    print(f"模擬資料：{N_SAMPLES} 位病人 x {N_FEATURES} 個特徵"
          f"（其中只有 {N_INFORMATIVE} 個是真正有用的）\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Linear Regression（無正則化）": LinearRegression(),
        "Ridge (L2, alpha=10)": Ridge(alpha=10.0),
        "Lasso (L1, alpha=0.5)": Lasso(alpha=0.5, max_iter=20000),
    }

    print(f"{'模型':32s} {'train R2':>10s} {'test R2':>10s} {'非零係數個數':>14s}")
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-6))
        print(f"{name:32s} {train_r2:10.3f} {test_r2:10.3f} {n_nonzero:14d}")

    print(f"\n真實情況：只有 {N_INFORMATIVE} / {N_FEATURES} 個特徵是有用的。")
    print("=> Linear Regression 在 train 上幾乎完美（train R2 接近 1），")
    print("   但 test R2 通常慘不忍睹甚至是負的 —— 這就是典型的『特徵數")
    print("   遠大於樣本數』overfitting。Ridge 只會把『所有』係數整體縮小，")
    print("   但這裡真正有用的特徵只有一小撮，Ridge 沒辦法把其餘 285 個")
    print("   雜訊特徵歸零，所以 test R2 依然不理想；Lasso 則會把大部分")
    print("   無用特徵的係數直接壓成 0（非零係數個數遠小於 300），使模型")
    print("   幾乎只用到『真正有用』的特徵，test R2 因此大幅提升。")
    print("   這正是 radiomics 分析中 Lasso 常被優先選用的原因。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 15 課）：
# 1) 說明 L1 與 L2 正則化在效果上的關鍵差異，尤其在「特徵篩選」這一點，
#    可以對照上面印出的『非零係數個數』來說明。
# 2) 調整 Lasso 的 alpha（例如 0.1、1.0、5.0），觀察非零係數個數與
#    test R2 如何隨 alpha 變化，alpha 太大或太小分別會發生什麼問題？
# ------------------------------------------------------------------
