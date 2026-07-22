"""
第 1 課：Hold-out 驗證法
資料集：Breast Cancer Wisconsin (Diagnostic) —— sklearn 內建，569 位病人、30 項腫瘤特徵、良性(0)/惡性(1)標籤

這支程式想讓你「親眼看到」單次 hold-out 切分的結果有多不穩定：
我們用 10 個不同的隨機種子各切一次 train/test，同一個模型、同一份資料，
準確率卻會忽高忽低 —— 這就是只做一次 hold-out 驗證的風險。
"""

import sys

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

RANDOM_SEEDS = list(range(10))


def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names


def run_single_holdout(X, y, random_state, stratify=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return accuracy_score(y_test, pred)


def main():
    X, y, _ = load_data()
    print(f"資料筆數: {len(y)}，惡性比例: {y.mean():.1%}\n")

    print("== 用 10 個不同的隨機種子各做一次 hold-out（80/20 切分）==")
    accs = []
    for seed in RANDOM_SEEDS:
        acc = run_single_holdout(X, y, random_state=seed, stratify=True)
        accs.append(acc)
        print(f"  random_state={seed:2d}  test accuracy = {acc:.4f}")

    accs = np.array(accs)
    print(f"\n10 次 hold-out 的準確率：mean={accs.mean():.4f}, "
          f"std={accs.std():.4f}, min={accs.min():.4f}, max={accs.max():.4f}")
    print("=> 光是換一個隨機種子，準確率就可能差好幾個百分點。")
    print("   如果論文只報告『其中一次』的結果，讀者根本不知道這個數字有多穩定。")

    print("\n== 對照：不做 stratify（不保持良性/惡性比例）的 hold-out ==")
    accs_no_strat = [run_single_holdout(X, y, random_state=s, stratify=False) for s in RANDOM_SEEDS]
    accs_no_strat = np.array(accs_no_strat)
    print(f"  不 stratify: mean={accs_no_strat.mean():.4f}, std={accs_no_strat.std():.4f}")
    print(f"  有 stratify: mean={accs.mean():.4f}, std={accs.std():.4f}")
    print("=> stratify 通常能讓不同切分之間的變異稍微小一點，尤其在類別不平衡時效果更明顯。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 1 課）：
# 1) 把 test_size 從 0.2 改成 0.5 再跑一次，觀察 std 是變大還是變小？為什麼？
# 2) 修改 run_single_holdout，改用 sklearn.tree.DecisionTreeClassifier 取代
#    LogisticRegression，比較兩個模型對「切分方式」的敏感程度是否不同。
# ------------------------------------------------------------------
