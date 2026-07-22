"""
第 10 課：Sensitivity / Specificity 與 ROC-AUC / PR-AUC
資料集：MNIST，沿用第 3、4 課『是 0 vs 不是 0』的極端不平衡設定（陽性僅 2%）

這一課：
  1) 計算 sensitivity（=recall）、specificity。
  2) 畫出 ROC curve 與 Precision-Recall curve，比較 ROC-AUC 與 PR-AUC。
  3) 具體示範：在極度不平衡資料下，ROC-AUC 可能『看起來很漂亮』，
     但 PR-AUC 才更誠實反映模型抓少數類別的能力。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity, specificity


def main():
    print("下載 / 讀取 MNIST 中 ...")
    X, y = load_imbalanced_mnist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"測試集: {len(y_test)} 筆，其中陽性(是0) = {y_test.sum()} 筆 "
          f"({y_test.mean():.2%})\n")

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    sens, spec = sensitivity_specificity(y_test, y_pred)
    print(f"預設閾值 0.5： sensitivity(recall) = {sens:.4f}，specificity = {spec:.4f}")

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"\nROC-AUC = {roc_auc:.4f}")
    print(f"PR-AUC  = {pr_auc:.4f}")

    # 對照組：一個完全隨機亂猜機率的『模型』
    rng = np.random.default_rng(0)
    y_prob_random = rng.uniform(0, 1, size=len(y_test))
    roc_auc_random = roc_auc_score(y_test, y_prob_random)
    pr_auc_random = average_precision_score(y_test, y_prob_random)
    print(f"\n【對照】完全隨機猜測： ROC-AUC = {roc_auc_random:.4f}"
          f"（理論值應接近 0.5），PR-AUC = {pr_auc_random:.4f}"
          f"（理論值應接近陽性比例 {y_test.mean():.4f}）")

    print("\n=> 觀察重點：即使模型隨便亂猜，因為負樣本(不是0)占絕大多數，")
    print("   ROC-AUC 依然會接近 0.5（不算差），但 PR-AUC 會非常接近陽性比例")
    print("   （在這裡大約 2%），PR-AUC 對『模型到底有沒有抓到少數類別』")
    print("   更誠實、更敏感。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0], name="Logistic Regression")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="隨機猜測")
    axes[0].set_title(f"ROC Curve (AUC={roc_auc:.3f})")
    axes[0].legend()

    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1], name="Logistic Regression")
    axes[1].axhline(y_test.mean(), linestyle="--", color="gray", label=f"隨機猜測基準線 ({y_test.mean():.3f})")
    axes[1].set_title(f"Precision-Recall Curve (PR-AUC={pr_auc:.3f})")
    axes[1].legend()

    fig.tight_layout()
    out_path = OUTPUT_DIR / "10_roc_pr_curve.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 10 課）：
# 1) 承第 4 課練習，掃描不同的決策閾值（0.1~0.9），計算每個閾值下的
#    sensitivity 與 specificity，畫出「sensitivity vs 1-specificity」，
#    確認畫出來的點會落在 ROC curve 上。
# 2) 把 POSITIVE_RATIO 改成 0.3（比較不極端），重新比較 ROC-AUC 與
#    PR-AUC 的差距是否縮小？這說明了不平衡程度如何影響兩個指標的差異。
# ------------------------------------------------------------------
