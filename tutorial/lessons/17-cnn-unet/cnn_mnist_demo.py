"""
第 17 課 (1/2)：CNN
資料集：MNIST（torchvision 自動下載）

訓練一個小型 CNN 做手寫數字十分類，跟第 16 課的 MLP 比較：
  - 參數量（CNN 靠權重共享，通常用更少參數達到更好效果）
  - Test accuracy
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loaders(batch_size=128, train_subset=6000, test_subset=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = MNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = MNIST(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False))


class SimpleMLP(nn.Module):
    """跟第 16 課同樣架構，當作比較基準。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """兩層卷積 + pooling，靠權重共享大幅減少參數量。"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28->14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14->7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, train_loader, test_loader, epochs=6, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(dim=1) == y).sum().item()
        acc = correct / len(test_loader.dataset)
        print(f"  epoch {epoch+1}/{epochs}  test_acc={acc:.4f}")
    return acc


def main():
    print(f"使用裝置: {DEVICE}")
    train_loader, test_loader = get_loaders()

    print("\n== MLP（第 16 課同款架構）==")
    mlp = SimpleMLP()
    print(f"參數量: {count_params(mlp):,}")
    mlp_acc = train_and_eval(mlp, train_loader, test_loader)

    print("\n== CNN（兩層卷積 + pooling）==")
    cnn = SimpleCNN()
    print(f"參數量: {count_params(cnn):,}")
    cnn_acc = train_and_eval(cnn, train_loader, test_loader)

    print(f"\n最終比較： MLP test_acc={mlp_acc:.4f} (參數 {count_params(mlp):,})")
    print(f"          CNN test_acc={cnn_acc:.4f} (參數 {count_params(cnn):,})")
    print("=> 在參數量相近的情況下，CNN 通常能拿到更高的準確率。這是因為")
    print("   卷積核在整張圖上『共用同一組權重』抓局部特徵（邊緣、線條），")
    print("   對圖像中的『平移』有一定程度的不變性，這種針對影像設計的")
    print("   歸納偏見（inductive bias）正是 MLP 沒有的優勢（第 20 課討論")
    print("   ViT 時還會再回來談這個概念）。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 17 課）：
# 1) 說明 CNN 的「權重共享」如何降低參數量並提升泛化能力，可以對照上面
#    印出的參數量數字來說明。
# 2) 在 SimpleCNN 中再加一層 Conv2d + ReLU + MaxPool2d，重新訓練，
#    觀察準確率與參數量的變化（記得同步調整 classifier 中 Linear 的
#    輸入維度）。
# 3) 接下來執行同資料夾的 unet_demo.py，看看把 CNN 改造成
#    encoder-decoder 架構（U-Net）之後，怎麼做『分割』而不是『分類』。
# ------------------------------------------------------------------
