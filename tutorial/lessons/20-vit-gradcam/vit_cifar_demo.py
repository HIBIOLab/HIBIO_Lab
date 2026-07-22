"""
第 20 課 (1/2)：Vision Transformer (ViT)
資料集：CIFAR-10（torchvision 自動下載，10 類小型彩色圖片，32x32）

訓練一個小型 CNN 和一個小型 ViT，在相同、有限的資料量與訓練時間下比較
準確率。ViT 缺乏 CNN 天生的『局部性 / 平移不變性』歸納偏見，在資料量小、
訓練不夠久的情況下通常會輸給 CNN —— 這正是 index.html 中提到的重點，
這裡讓你親眼驗證。
"""

import sys
import time
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 32
PATCH_SIZE = 4
N_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 8x8 = 64 patches
D_MODEL = 64


def get_loaders(batch_size=128, train_subset=8000, test_subset=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False))


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16->8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class TinyViT(nn.Module):
    """把影像切成 4x4 patch，攤平投影成 token，加上 [CLS] token + 位置編碼，
    餵進標準 Transformer encoder。"""

    def __init__(self, n_classes=10, depth=2, n_heads=4):
        super().__init__()
        patch_dim = PATCH_SIZE * PATCH_SIZE * 3
        self.patch_embed = nn.Linear(patch_dim, D_MODEL)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES + 1, D_MODEL) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=n_heads, dim_feedforward=D_MODEL * 2,
            batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(D_MODEL, n_classes)

    def to_patches(self, x):
        # x: (B, 3, 32, 32) -> (B, N_PATCHES, patch_dim)
        B, C, H, W = x.shape
        p = PATCH_SIZE
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B,C,H/p,W/p,p,p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, N_PATCHES, C * p * p)
        return x

    def forward(self, x):
        patches = self.to_patches(x)
        tokens = self.patch_embed(patches)  # (B, N_PATCHES, D_MODEL)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed

        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0, :]  # 用 [CLS] token 的輸出做分類
        return self.head(cls_out)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, train_loader, test_loader, epochs=10, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    start = time.time()
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
    return acc, time.time() - start


def main():
    print(f"使用裝置: {DEVICE}")
    train_loader, test_loader = get_loaders()
    print(f"訓練集: {len(train_loader.dataset)} 張，測試集: {len(test_loader.dataset)} 張\n")

    print("== SmallCNN ==")
    cnn = SmallCNN()
    print(f"參數量: {count_params(cnn):,}")
    cnn_acc, cnn_time = train_and_eval(cnn, train_loader, test_loader)

    print("\n== TinyViT（patch_size=4, 2 層 Transformer encoder）==")
    vit = TinyViT()
    print(f"參數量: {count_params(vit):,}")
    vit_acc, vit_time = train_and_eval(vit, train_loader, test_loader)

    print(f"\n最終比較（訓練資料只有 {len(train_loader.dataset)} 張，訓練 10 epoch）：")
    print(f"  CNN: test_acc={cnn_acc:.4f}  參數={count_params(cnn):,}  耗時={cnn_time:.1f}s")
    print(f"  ViT: test_acc={vit_acc:.4f}  參數={count_params(vit):,}  耗時={vit_time:.1f}s")
    if cnn_acc > vit_acc:
        print("\n=> 在資料量有限、訓練沒有很久的情況下，CNN 通常會贏過從頭訓練的")
        print("   ViT。CNN 的卷積 + pooling 天生假設『局部性』與『平移不變性』，")
        print("   這種歸納偏見讓它在小資料集上更快學到有用的特徵；ViT 幾乎要")
        print("   從資料中重新學這些先驗知識，因此通常需要更大量的資料或")
        print("   大規模預訓練（如 ImageNet-21k）才能追上甚至超越 CNN。")
    else:
        print("\n=> 這次 ViT 表現不輸 CNN，但這通常需要足夠的資料量與訓練時間；")
        print("   資料量更小或訓練更短時，ViT 通常會相對吃虧。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 20 課）：
# 1) 把 train_subset 從 8000 調大到 40000，重新比較 CNN 與 ViT 的差距
#    是否縮小？這驗證了『資料量越大，ViT 的劣勢越不明顯』這個論點。
# 2) 為什麼 ViT 需要 self.pos_embed，而 CNN 不需要類似的東西？
#    （提示：回顧第 19 課 attention 為什麼需要位置編碼）
# ------------------------------------------------------------------
