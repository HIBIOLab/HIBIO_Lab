"""
第 4 課：Precision, Recall, F1-score
資料集：MNIST（手寫數字），沿用第 3 課「是 0 vs 不是 0」的極端不平衡設定（陽性僅 2%）

這一課用同樣的資料，比較「無腦模型」「一般 Logistic Regression」「加上
class_weight='balanced' 的 Logistic Regression」三者的 precision / recall /
F1，具體看到「重視 recall」會讓 precision 付出什麼代價。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TARGET_DIGIT = 0
POSITIVE_RATIO = 0.02


def load_imbalanced_mnist():
    train_set = MNIST(root=str(DATA_DIR), train=True, download=True)
    X = train_set.data.numpy().reshape(len(train_set), -1) / 255.0
    y = (train_set.targets.numpy() == TARGET_DIGIT).astype(int)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng = np.random.default_rng(42)
    n_neg = 4000
    n_pos = int(n_neg * POSITIVE_RATIO / (1 - POSITIVE_RATIO))
    pos_idx = rng.choice(pos_idx, size=n_pos, replace=False)
    neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return X[idx], y[idx]


def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name:32s} accuracy={acc:.4f}  precision={prec:.4f}  "
          f"recall={rec:.4f}  f1={f1:.4f}")
    return acc, prec, rec, f1


def main():
    print("下載 / 讀取 MNIST 中 ...")
    X, y = load_imbalanced_mnist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"測試集大小={len(y_test)}，其中陽性(是0)={y_test.sum()} 筆\n")

    print(f"{'模型':32s} {'accuracy':>10s}  {'precision':>10s}  {'recall':>8s}  {'f1':>6s}")

    naive_pred = np.zeros_like(y_test)
    evaluate("無腦模型（永遠猜『不是0』）", y_test, naive_pred)

    model_plain = LogisticRegression(max_iter=2000)
    model_plain.fit(X_train, y_train)
    pred_plain = model_plain.predict(X_test)
    evaluate("Logistic Regression（一般）", y_test, pred_plain)

    model_balanced = LogisticRegression(max_iter=2000, class_weight="balanced")
    model_balanced.fit(X_train, y_train)
    pred_balanced = model_balanced.predict(X_test)
    evaluate("Logistic Regression（class_weight=balanced）", y_test, pred_balanced)

    print("\n=> 觀察重點：class_weight='balanced' 通常會讓 recall 明顯上升")
    print("   （抓到更多真正的『0』），但代價是 precision 下降（更多假警報）。")
    print("   這正是『重視漏診 vs 重視假警報』的權衡，實務上要依臨床情境決定。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 4 課）：
# 1) 試著調整 LogisticRegression 的決策閾值（predict_proba 搭配自訂
#    threshold，而不是預設的 0.5），畫出 threshold 從 0.1 到 0.9 時
#    precision / recall 如何此消彼長。
# 2) 承第 3 課練習：把 POSITIVE_RATIO 改成 0.3，重新比較三個模型的
#    precision/recall/f1，不平衡程度改變後，class_weight='balanced'
#    帶來的差異是變大還是變小？
# ------------------------------------------------------------------
