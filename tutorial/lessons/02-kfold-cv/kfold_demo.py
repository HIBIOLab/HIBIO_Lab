"""
第 2 課：5-fold Cross-Validation
資料集：Breast Cancer Wisconsin (Diagnostic) —— sklearn 內建

延續第 1 課的觀察：單次 hold-out 的分數不穩定。這一課改用 K-fold，
把資料切成 K 份，輪流當一次驗證集，最後看「平均值 ± 標準差」，
而不是只看單一數字。
"""

import sys

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

    print("== 不同 K 值下的 cross-validation 結果比較 ==")
    for k in (2, 3, 5, 10):
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"K={k:2d}: 各 fold 分數 = {np.round(scores, 4)}")
        print(f"        mean={scores.mean():.4f}  std={scores.std():.4f}")

    print("\n== 5-fold 的完整流程（手動展開，方便理解每一步發生什麼事）==")
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv5.split(X, y), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        fold_scores.append(score)

        print(f"  Fold {fold_idx}: train={len(train_idx)} 筆, val={len(val_idx)} 筆, "
              f"val 中惡性比例={y_val.mean():.1%}, accuracy={score:.4f}")

    fold_scores = np.array(fold_scores)
    print(f"\n5-fold 最終報告：{fold_scores.mean():.4f} ± {fold_scores.std():.4f}")
    print("=> 論文 / 報告應該寫這個「平均值 ± 標準差」，而不是任選一個 fold 的分數。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 2 課）：
# 1) 把上面手動展開的 StratifiedKFold 換成 sklearn.model_selection.
#    StratifiedGroupKFold，自己造一組假的病人 ID（例如每 3 筆資料視為
#    同一位病人），確保同一病人不會同時出現在 train 與 val。
#    （這是為第 9 課「病人層級切分」暖身的練習）
# 2) 比較 K=2 和 K=10 的 std，何者比較小？這跟每個 fold 的驗證集大小
#    有什麼關係？
# ------------------------------------------------------------------
