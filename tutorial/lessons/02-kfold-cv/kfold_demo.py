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

# ==================================================================
# 課後練習（對照 index.html 第 2 課）
# ==================================================================
#
# 練習 1：K 值大小如何影響「分數的穩定度」（std）
# ------------------------------------------------------------------
#   目標：親手驗證「K 越大，std 不一定越小」這件事，並理解背後原因。
#
#   做法：
#     - 看上面「不同 K 值」那段輸出，把 K=2 和 K=10 的 std 抄下來比較。
#     - 想一下：K 越大 → 每個 fold 的「驗證集」越小（K=10 時每份只有
#       約 1/10 資料當驗證），而「小驗證集」的分數本身就比較容易因為
#       運氣而跳動，所以單看某一個 fold 會更不穩；但因為考了 10 次再
#       平均，「平均值」反而通常更接近真實水準。
#
#   觀察重點（可以自己加幾行程式印出來確認）：
#     - K 小（K=2）：每個 fold 驗證集大、單一分數穩，但只考 2 次，
#       平均值的代表性較弱，而且拿來訓練的資料只剩一半。
#     - K 大（K=10）：每個 fold 驗證集小、各 fold 分數起伏可能較大，
#       但平均值更可靠；缺點是要訓練 10 次、比較花時間。
#     - 結論：K 通常取 5 或 10 是「訓練資料量 / 計算成本 / 估計穩定度」
#       三者之間的折衷，不是越大越好。
#
#
# 練習 2：把 StratifiedKFold 換成 StratifiedGroupKFold（病人層級切分暖身）
# ------------------------------------------------------------------
#   目標：先體會「同一位病人的資料若同時出現在 train 與 val，分數會被
#   高估」這個陷阱，這正是第 7 課 data leakage 的核心。
#
#   做法：
#     1) 造一組假的病人 ID：把每 3 筆資料當成同一位病人，例如
#            groups = np.arange(len(y)) // 3
#        （真實情況是「一位病人有多張切片 / 多次檢查」，這裡用假 ID 模擬。）
#     2) 匯入並改用：
#            from sklearn.model_selection import StratifiedGroupKFold
#            cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
#        切分時記得把 groups 傳進去：cv.split(X, y, groups)
#     3) 在每個 fold 裡驗證「train 和 val 的病人完全沒有重疊」，例如：
#            assert set(groups[train_idx]).isdisjoint(set(groups[val_idx]))
#
#   觀察重點：
#     - StratifiedGroupKFold 會「同時」照顧兩件事：類別比例維持一致
#       （Stratified）、而且同一病人不會被拆到兩邊（Group）。
#     - 對照原本的 StratifiedKFold（沒有 group 概念），想想看：如果同
#       一病人的多張切片彼此很像，卻一部分在 train、一部分在 val，模型
#       等於「考試前偷看過答案」，測出來的分數會偏高、不可信。
#     - 醫學影像資料集常常只有幾十到幾百位病人，論文標準做法就是用
#       StratifiedGroupKFold 做 5-fold / 10-fold，而不是單純的 KFold。
# ==================================================================
