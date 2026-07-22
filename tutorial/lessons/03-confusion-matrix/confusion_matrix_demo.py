"""
第 3 課：Confusion Matrix 與 Accuracy 的陷阱
資料集：MNIST（手寫數字），torchvision 自動下載

我們把問題改造成「這張圖是不是數字 0？」的二元分類，並且刻意把「是 0」的
樣本壓到只剩 2%，模擬醫學篩檢中「陽性樣本很少」的情境（例如腫瘤篩檢）。
接著比較「無腦模型（永遠猜不是0）」跟「真正訓練的模型」的 accuracy 與
confusion matrix，親眼看看 accuracy 有多容易騙人。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TARGET_DIGIT = 0
POSITIVE_RATIO = 0.02  # 刻意把「是 0」的比例壓到 2%，模擬醫學篩檢的不平衡


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


def print_confusion(cm, title):
    # cm 是用 labels=[1, 0] 算出來的，所以 ravel() 順序是 [TP, FN, FP, TN]
    # （row0=actual"1", row1=actual"0"；col0=predicted"1", col1=predicted"0"）
    tp, fn, fp, tn = cm.ravel()
    print(f"\n{title}")
    print("                 predicted 0    predicted not-0")
    print(f"  actual 0        TP={tp:<5d}      FN={fn:<5d}")
    print(f"  actual not-0    FP={fp:<5d}      TN={tn:<5d}")
    return tn, fp, fn, tp


def main():
    print("下載 / 讀取 MNIST 中 ...")
    X, y = load_imbalanced_mnist()
    print(f"資料筆數: {len(y)}，其中『是 0』的比例: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 1) 無腦基準模型：永遠猜「不是 0」
    naive_pred = np.zeros_like(y_test)
    naive_acc = accuracy_score(y_test, naive_pred)
    cm_naive = confusion_matrix(y_test, naive_pred, labels=[1, 0])
    print_confusion(cm_naive, f"【無腦模型：永遠猜『不是 0』】 accuracy = {naive_acc:.4f}")
    print("=> 完全沒有偵測出任何一個真正的 0，但 accuracy 卻高達",
          f"{naive_acc:.1%}！這就是 accuracy 在不平衡資料下的陷阱。")

    # 2) 真的訓練一個模型
    model = LogisticRegression(max_iter=2000, class_weight=None)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[1, 0])
    print_confusion(cm, f"【真正訓練的 Logistic Regression】 accuracy = {acc:.4f}")
    print("=> accuracy 看起來可能跟無腦模型差不多甚至更低一點點，")
    print("   但這個模型『真的』抓到了一些 0，代表 accuracy 不能單獨拿來判斷模型好壞。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 3 課）：
# 1) 把 POSITIVE_RATIO 改成 0.3（不那麼不平衡），重新跑一次，比較無腦模型
#    與訓練模型的 accuracy 差距是變大還是變小？
# 2) 把 LogisticRegression 換成 class_weight="balanced"，觀察 confusion
#    matrix 中 TP/FN 的變化（下一課會用 precision/recall 量化這個變化）。
# ------------------------------------------------------------------
