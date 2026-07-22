"""
第 16 課：Multi-Layer Perceptron (MLP)
資料集：MNIST（torchvision 自動下載）

訓練一個簡單的 MLP（784 -> 128 -> 64 -> 10）做手寫數字十分類，並且：
  1) 比較「有 activation function」vs「全部拿掉、只剩線性層」的差異，
     具體驗證『沒有非線性，多層網路退化成線性模型』這件事。
  2) 比較「有 dropout」vs「沒有 dropout」在訓練較久之後的過擬合程度。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(batch_size=128, train_subset=6000, test_subset=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = MNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = MNIST(root=str(DATA_DIR), train=False, download=True, transform=tfm)

    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class MLP(nn.Module):
    def __init__(self, use_activation=True, use_dropout=False, dropout_p=0.5):
        super().__init__()
        act = nn.ReLU() if use_activation else nn.Identity()
        layers = [
            nn.Flatten(),
            nn.Linear(28 * 28, 128), act,
        ]
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers += [nn.Linear(128, 64), act]
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        layers += [nn.Linear(64, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model, train_loader, test_loader, epochs=8, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    def evaluate(loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
        return correct / len(loader.dataset)

    train_losses, train_accs, test_accs = [], [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_losses.append(total_loss / len(train_loader.dataset))

        train_acc = evaluate(train_loader)
        test_acc = evaluate(test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"  epoch {epoch+1}/{epochs}  train_loss={train_losses[-1]:.4f}  "
              f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}  "
              f"gap={train_acc - test_acc:+.4f}")

    return train_losses, train_accs, test_accs


def main():
    print(f"使用裝置: {DEVICE}")
    train_loader, test_loader = get_loaders()

    print("\n== (1) 有 ReLU activation vs 全部換成 Identity（等於純線性模型）==")
    print("-- 有 activation --")
    model_with_act = MLP(use_activation=True)
    _, _, acc_with_act = train_model(model_with_act, train_loader, test_loader, epochs=5)

    print("-- 沒有 activation（三層線性層疊在一起）--")
    model_no_act = MLP(use_activation=False)
    _, _, acc_no_act = train_model(model_no_act, train_loader, test_loader, epochs=5)

    print(f"\n最終 test accuracy： 有 activation={acc_with_act[-1]:.4f}   "
          f"沒有 activation={acc_no_act[-1]:.4f}")
    print("=> 拿掉非線性 activation 後，數學上『Linear -> Linear -> Linear』")
    print("   直接乘起來還是一個 Linear 轉換，等價於單層的線性分類器。")
    print("   在 MNIST 這種相對簡單的任務、訓練 epoch 不多時，差距可能不算")
    print("   誇張，但方向是一致的：加了非線性通常比較好、也比較穩定。")
    print("   資料越複雜或網路越深，這個差距通常會更明顯（見課後練習）。")

    print("\n== (2) Dropout 對過擬合程度的影響（刻意用較小的訓練集 2000 筆，")
    print("   訓練較久的 25 epoch，讓 train/test 的差距更容易被看到）==")
    small_train_loader, _ = get_loaders(train_subset=2000)

    print("-- 沒有 dropout --")
    model_no_dropout = MLP(use_activation=True, use_dropout=False)
    _, train_accs_nd, test_accs_nd = train_model(
        model_no_dropout, small_train_loader, test_loader, epochs=25)

    print("-- 有 dropout(p=0.5) --")
    model_dropout = MLP(use_activation=True, use_dropout=True, dropout_p=0.5)
    _, train_accs_d, test_accs_d = train_model(
        model_dropout, small_train_loader, test_loader, epochs=25)

    gap_nd = train_accs_nd[-1] - test_accs_nd[-1]
    gap_d = train_accs_d[-1] - test_accs_d[-1]
    print(f"\n最後一個 epoch 的 train-test 準確率差距：")
    print(f"  沒有 dropout: train={train_accs_nd[-1]:.4f}  test={test_accs_nd[-1]:.4f}  gap={gap_nd:.4f}")
    print(f"  有 dropout  : train={train_accs_d[-1]:.4f}  test={test_accs_d[-1]:.4f}  gap={gap_d:.4f}")
    print("=> train 與 test 準確率之間的差距就是『過擬合程度』的具體指標：")
    print("   差距越大，代表模型越是在『背訓練資料』而不是學到通用的樣態。")
    print("   dropout 訓練時隨機關閉神經元，通常能讓這個差距縮小。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    axes[0].plot(train_accs_nd, label="train accuracy", marker="o")
    axes[0].plot(test_accs_nd, label="test accuracy", marker="o")
    axes[0].set_title("沒有 Dropout")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(train_accs_d, label="train accuracy", marker="o")
    axes[1].plot(test_accs_d, label="test accuracy", marker="o")
    axes[1].set_title("有 Dropout (p=0.5)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.suptitle("Train / Test Accuracy 差距 = 過擬合程度")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "16_mlp_dropout.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 16 課）：
# 1) 為什麼神經網路需要非線性 activation function？結合上面的實驗結果，
#    用自己的話解釋「全部換成 Identity」為什麼準確率會下降。
# 2) 把 train_subset 從 6000 調大到 20000（訓練資料變多），重新比較
#    有無 dropout 的差異是否縮小？這跟『資料量越多、越不容易 overfit』
#    的觀念是否一致？
# ------------------------------------------------------------------
