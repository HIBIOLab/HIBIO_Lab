"""
第 20 課 (2/2)：Grad-CAM —— 讓 CNN 的判斷『看得見』
資料集：CIFAR-10（torchvision 自動下載）

訓練一個小型 CNN 做 CIFAR-10 分類，再用 Grad-CAM 找出『模型在看哪裡』：
利用最後一層卷積特徵圖對預測分數的梯度，反推出對這次預測貢獻最大的
區域，畫成熱力圖疊在原圖上 —— 這是讓臨床醫師能檢視模型判斷依據、
建立信任的關鍵工具之一。
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def get_loaders(batch_size=128, train_subset=8000, test_subset=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False))


class SmallCNN(nn.Module):
    """跟 vit_cifar_demo.py 同款架構；self.features 的最後一層卷積輸出
    就是 Grad-CAM 要抓的目標層。"""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())  # <- Grad-CAM 目標層
        self.pool2 = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        feat = self.conv2(x)
        x = self.pool2(feat)
        out = self.classifier(x)
        return out, feat  # 同時回傳最後一層卷積的特徵圖，供 Grad-CAM 使用


def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out, _ = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out, _ = model(x)
                correct += (out.argmax(dim=1) == y).sum().item()
        print(f"  epoch {epoch+1}/{epochs}  test_acc={correct/len(test_loader.dataset):.4f}")


def grad_cam(model, image, target_class=None):
    """回傳 (predicted_class, cam_heatmap)。cam_heatmap 大小和最後一層
    卷積特徵圖相同（這裡是 8x8），會再被放大回原圖大小方便疊圖。"""
    model.eval()
    image = image.unsqueeze(0).to(DEVICE).requires_grad_(False)

    out, feat = model(image)  # feat: (1, C, 8, 8)，requires_grad 由 model 參數帶出
    feat.retain_grad()

    if target_class is None:
        target_class = out.argmax(dim=1).item()

    score = out[0, target_class]
    model.zero_grad()
    score.backward()

    grads = feat.grad[0]          # (C, 8, 8)
    weights = grads.mean(dim=(1, 2))  # 對每個 channel 做全域平均池化，得到重要性權重

    cam = torch.zeros(feat.shape[2:], device=DEVICE)
    for c, w in enumerate(weights):
        cam += w * feat[0, c].detach()
    cam = F.relu(cam)  # 只保留對預測有正向貢獻的區域
    cam = cam / (cam.max() + 1e-8)

    # 把 8x8 的熱力圖放大回 32x32，方便疊在原圖上
    cam_resized = F.interpolate(
        cam.view(1, 1, *cam.shape), size=(32, 32), mode="bilinear", align_corners=False
    )[0, 0]
    return target_class, cam_resized.cpu().numpy()


def main():
    print(f"使用裝置: {DEVICE}")
    train_loader, test_loader = get_loaders()

    print("== 訓練 SmallCNN ==")
    model = SmallCNN()
    train_model(model, train_loader, test_loader)

    print("\n== 對測試集中的幾張圖做 Grad-CAM ==")
    n_show = 6
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5))

    for i in range(n_show):
        img, label = test_loader.dataset[i]
        pred_class, cam_resized = grad_cam(model, img)
        img_np = img.permute(1, 2, 0).numpy()

        axes[0, i].imshow(img_np)
        axes[0, i].set_title(f"真實:{CLASS_NAMES[label]}\n預測:{CLASS_NAMES[pred_class]}", fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(img_np)
        axes[1, i].imshow(cam_resized, cmap="jet", alpha=0.5)
        axes[1, i].set_title("Grad-CAM", fontsize=9)
        axes[1, i].axis("off")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "20_gradcam.png"
    fig.savefig(out_path, dpi=120)
    print(f"圖片已存到: {out_path}")
    print("=> 熱力圖越亮（紅/黃）代表該區域對這次預測的貢獻越大。")
    print("   如果模型是因為背景雜訊而不是動物本身做出判斷，Grad-CAM 通常")
    print("   會亮在不合理的區域，這就是這個工具能幫忙抓出模型『學錯重點』")
    print("   的原因，在醫學影像上尤其重要：要確認模型是真的在看病灶，")
    print("   而不是影像邊角的標記或掃描機台特徵。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 20 課）：
# 1) grad_cam() 中 `weights = grads.mean(dim=(1, 2))` 這一步在做什麼？
#    為什麼要對每個 channel 的梯度做「全域平均池化」？
# 2) 找一張模型『預測錯誤』的測試圖片，畫出它的 Grad-CAM，觀察模型是不是
#    真的看錯地方了，還是圖片本身就容易混淆（例如貓跟狗的輪廓很像）。
# ------------------------------------------------------------------
