"""
第 16 課：Multi-Layer Perceptron (MLP)
資料集：Iris（sklearn 內建 load_iris）—— 150 筆、4 個花萼/花瓣特徵、3 個品種

用一個簡單的 MLP（4 -> 32 -> 16 -> 3）做鳶尾花三分類，並且：
  1) 比較「有 activation function」vs「全部拿掉、只剩線性層」的差異，
     具體驗證『沒有非線性，多層網路其實退化成一個線性模型』這件事。
  2) 比較「有 dropout」vs「沒有 dropout」在『刻意只用很少訓練資料、
     訓練很久』時的過擬合程度。

備註：Iris 是純表格資料（4 個數值特徵），這一課的重點是把 MLP 的兩個核心
觀念（非線性 activation、dropout）用最小、最快的資料集講清楚；影像資料留到
第 17 課的 CNN / U-Net 再登場。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FEATURES = 4   # Iris 有 4 個特徵
N_CLASSES = 3    # 3 個品種


def get_tensors(train_size=0.7, seed=42):
    """載入 Iris、標準化、切成 train/test，回傳成 torch tensor。"""
    data = load_iris()
    X, y = data.data.astype(np.float32), data.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, stratify=y, random_state=seed)

    # StandardScaler 只 fit 在 train 上（避免第 7 課講過的資料洩漏）
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    to_t = lambda a: torch.from_numpy(a)
    return (to_t(X_train), to_t(y_train), to_t(X_test), to_t(y_test))


class MLP(nn.Module):
    def __init__(self, hidden=(32, 16), use_activation=True, use_dropout=False, dropout_p=0.5):
        super().__init__()
        act = nn.ReLU() if use_activation else nn.Identity()
        layers = []
        in_dim = N_FEATURES
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act)
            if use_dropout:
                layers.append(nn.Dropout(dropout_p))
            in_dim = h
        layers.append(nn.Linear(in_dim, N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, X_train, y_train, X_test, y_test, epochs=200, lr=1e-2):
    model.to(DEVICE)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def accuracy(X, y):
        model.eval()
        with torch.no_grad():
            pred = model(X).argmax(dim=1)
        return (pred == y).float().mean().item()

    train_accs, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()

        train_accs.append(accuracy(X_train, y_train))
        test_accs.append(accuracy(X_test, y_test))
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  loss={loss.item():.4f}  "
                  f"train_acc={train_accs[-1]:.4f}  test_acc={test_accs[-1]:.4f}")

    return train_accs, test_accs


def main():
    print(f"使用裝置: {DEVICE}")
    X_train, y_train, X_test, y_test = get_tensors()
    print(f"訓練集 {len(X_train)} 筆、測試集 {len(X_test)} 筆，各 {N_FEATURES} 個特徵、{N_CLASSES} 類\n")

    print("== (1) 有 ReLU activation vs 全部換成 Identity（等於純線性模型）==")
    torch.manual_seed(0)
    print("-- 有 activation --")
    _, acc_with_act = train_model(MLP(use_activation=True), X_train, y_train, X_test, y_test)

    torch.manual_seed(0)
    print("-- 沒有 activation（多層線性層疊在一起）--")
    _, acc_no_act = train_model(MLP(use_activation=False), X_train, y_train, X_test, y_test)

    print(f"\n最終 test accuracy： 有 activation={acc_with_act[-1]:.4f}   "
          f"沒有 activation={acc_no_act[-1]:.4f}")
    print("=> 拿掉非線性 activation 後，數學上『Linear -> Linear -> Linear』乘起來")
    print("   還是一個 Linear 轉換，等價於單層的線性分類器。Iris 本身接近線性")
    print("   可分，所以純線性模型也能有不錯的成績，但加了非線性通常更穩、")
    print("   在複雜資料上差距會更明顯（見課後練習）。")

    print("\n== (2) Dropout 對過擬合程度的影響（刻意只用很少訓練資料、訓練很久，")
    print("   讓 train/test 的差距更容易被看到）==")
    # 只取 24 筆當訓練資料，並把網路加寬，人為製造容易過擬合的情境
    small_n = 24
    Xs, ys = X_train[:small_n], y_train[:small_n]
    wide = (128, 128)

    print(f"-- 沒有 dropout（訓練資料僅 {small_n} 筆、隱藏層 {wide}）--")
    torch.manual_seed(0)
    train_nd, test_nd = train_model(
        MLP(hidden=wide, use_dropout=False), Xs, ys, X_test, y_test, epochs=400)

    print(f"-- 有 dropout(p=0.5) --")
    torch.manual_seed(0)
    train_d, test_d = train_model(
        MLP(hidden=wide, use_dropout=True, dropout_p=0.5), Xs, ys, X_test, y_test, epochs=400)

    gap_nd = train_nd[-1] - test_nd[-1]
    gap_d = train_d[-1] - test_d[-1]
    print(f"\n最後一個 epoch 的 train-test 準確率差距：")
    print(f"  沒有 dropout: train={train_nd[-1]:.4f}  test={test_nd[-1]:.4f}  gap={gap_nd:.4f}")
    print(f"  有 dropout  : train={train_d[-1]:.4f}  test={test_d[-1]:.4f}  gap={gap_d:.4f}")
    print("=> train 與 test 準確率的差距就是『過擬合程度』的具體指標：差距越大，")
    print("   代表模型越是在『背訓練資料』。dropout 訓練時隨機關閉神經元，通常")
    print("   能讓這個差距縮小。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].plot(train_nd, label="train accuracy")
    axes[0].plot(test_nd, label="test accuracy")
    axes[0].set_title("沒有 Dropout")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(train_d, label="train accuracy")
    axes[1].plot(test_d, label="test accuracy")
    axes[1].set_title("有 Dropout (p=0.5)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle("Train / Test Accuracy 差距 = 過擬合程度（Iris，僅 24 筆訓練資料）")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "16_mlp_dropout.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 16 課）：
# 1) 為什麼神經網路需要非線性 activation function？結合上面的實驗結果，
#    用自己的話解釋「全部換成 Identity」為什麼在複雜資料上會退化成線性模型。
#    （提示：Iris 幾乎線性可分，所以差距不大；試著換成 sklearn 的
#    load_digits 或 make_moons 這類較難、非線性的資料，差距會更明顯。）
# 2) 把 (2) 的 small_n 從 24 調大到 120（訓練資料變多），重新比較有無
#    dropout 的差異是否縮小？這跟『資料量越多、越不容易 overfit』的觀念
#    是否一致？
# ------------------------------------------------------------------
