"""
第 20 課 (2/2)：Grad-CAM —— 讓 CNN 的判斷『看得見』
資料集：COVID-19 Chest X-ray（胸部 X 光三分類：Covid / Normal / Viral Pneumonia）
        放在 tutorial/data/Covid19-classification/（train/ 與 test/，各含三個類別資料夾）
        來源：https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

訓練一個小型 CNN 做胸部 X 光三分類，再用 Grad-CAM 找出『模型在看哪裡』：
利用最後一層卷積特徵圖對預測分數的梯度，反推出對這次預測貢獻最大的區域，
畫成熱力圖疊在原始 X 光上 —— 在醫學影像上這正是關鍵：要確認模型是真的在
看肺部病灶，而不是影像邊角的標記、骨架或掃描機台特徵。

（相較於 CIFAR-10 的 32x32 小圖，這裡用較大的胸部 X 光，Grad-CAM 熱力圖
清楚很多，也更貼近臨床上「模型可解釋性」真正在乎的問題。）
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
from PIL import Image
from torchvision.transforms import functional as VF

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COVID_DIR = DATA_DIR / "Covid19-classification"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["Covid", "Normal", "Viral Pneumonia"]
IMG_SIZE = 128


def load_split(split):
    """讀 Covid19-classification/<split> 三個類別的所有胸部 X 光（灰階、縮放、0-1）。"""
    split_dir = COVID_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"找不到 {split_dir}。請確認 COVID-19 影像資料集已放在 {COVID_DIR}（含 train/ 與 test/）。")
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((split_dir / cls).glob("*")):
            img = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            xs.append(np.asarray(img, dtype=np.float32) / 255.0)
            ys.append(label)
    X = torch.from_numpy(np.stack(xs)).unsqueeze(1)   # (N, 1, H, W)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


class SmallCNN(nn.Module):
    """三層卷積的小型 CNN（1 通道灰階進、3 類出）。
    self.conv3 的輸出（32x32 特徵圖）就是 Grad-CAM 要抓的目標層。"""

    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)   # 128 -> 64
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)   # 64 -> 32
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())  # <- Grad-CAM 目標層
        self.pool3 = nn.MaxPool2d(2)   # 32 -> 16
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        feat = self.conv3(x)          # (B, 64, 32, 32)
        x = self.pool3(feat)
        out = self.classifier(x)
        return out, feat              # 同時回傳最後一層卷積特徵圖，供 Grad-CAM 使用


def safe_augment(xb):
    """訓練時的安全增強（小角度旋轉 + 亮度擾動），減輕小資料集的過擬合。"""
    angle = float(torch.empty(1).uniform_(-8, 8))
    xb = VF.rotate(xb, angle)
    gamma = float(torch.empty(1).uniform_(0.8, 1.3))
    return torch.clamp(xb, 0, 1) ** gamma


def train_model(model, X_train, y_train, X_test, y_test, epochs=30, lr=1e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xte, yte = X_test.to(DEVICE), y_test.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 16):
            idx = perm[i:i + 16]
            xb, yb = safe_augment(X_train[idx]).to(DEVICE), y_train[idx].to(DEVICE)
            opt.zero_grad()
            out, _ = model(xb)
            loss_fn(out, yb).backward()
            opt.step()
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte)[0].argmax(1) == yte).float().mean().item()
            print(f"  epoch {epoch+1:2d}/{epochs}  test_acc={acc:.4f}")


def grad_cam(model, image, target_class=None):
    """回傳 (predicted_class, cam_heatmap)。cam 與最後一層卷積特徵圖同大小（32x32），
    再放大回原圖大小方便疊圖。"""
    model.eval()
    image = image.unsqueeze(0).to(DEVICE)

    out, feat = model(image)          # feat: (1, C, 32, 32)
    feat.retain_grad()
    if target_class is None:
        target_class = out.argmax(dim=1).item()

    model.zero_grad()
    out[0, target_class].backward()

    weights = feat.grad[0].mean(dim=(1, 2))   # 每個 channel 的梯度做全域平均池化 = 重要性權重
    cam = torch.zeros(feat.shape[2:], device=DEVICE)
    for c, w in enumerate(weights):
        cam += w * feat[0, c].detach()
    cam = F.relu(cam)                          # 只保留對預測有正向貢獻的區域
    cam = cam / (cam.max() + 1e-8)

    cam_resized = F.interpolate(
        cam.view(1, 1, *cam.shape), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )[0, 0]
    return target_class, cam_resized.cpu().numpy()


def main():
    print(f"使用裝置: {DEVICE}")
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    print(f"訓練 {len(X_train)} 張、測試 {len(X_test)} 張胸部 X 光（{len(CLASSES)} 類）\n")

    print("== 訓練 SmallCNN ==")
    torch.manual_seed(0)
    model = SmallCNN()
    train_model(model, X_train, y_train, X_test, y_test)

    # 每個類別各挑 2 張測試圖來畫 Grad-CAM
    show_idx = []
    for label in range(len(CLASSES)):
        show_idx += (y_test == label).nonzero().flatten().tolist()[:2]

    print("\n== 對測試集的幾張 X 光做 Grad-CAM ==")
    n = len(show_idx)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5))
    for col, i in enumerate(show_idx):
        img, label = X_test[i], int(y_test[i])
        pred, cam = grad_cam(model, img)
        img_np = img[0].numpy()

        axes[0, col].imshow(img_np, cmap="gray")
        axes[0, col].set_title(f"真實:{CLASSES[label]}\n預測:{CLASSES[pred]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(img_np, cmap="gray")
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].set_title("Grad-CAM", fontsize=8)
        axes[1, col].axis("off")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "20_gradcam.png"
    fig.savefig(out_path, dpi=120)
    print(f"圖片已存到: {out_path}")
    print("=> 熱力圖越亮（紅/黃）代表該區域對這次預測的貢獻越大。理想情況下，模型")
    print("   對 Covid / Viral Pneumonia 的判斷應該亮在『肺部』區域；如果亮在影像")
    print("   邊角的標記、肩膀骨架或機台文字上，就代表模型可能學到了『捷徑』而不是")
    print("   真正的病理特徵——這正是 Grad-CAM 在醫學影像上不可或缺的原因。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 20 課）：
# 1) grad_cam() 中 `weights = feat.grad[0].mean(dim=(1, 2))` 這一步在做什麼？
#    為什麼要對每個 channel 的梯度做「全域平均池化」？
# 2) 找一張模型『預測錯誤』的測試 X 光，畫出它的 Grad-CAM，觀察模型是不是
#    看錯地方了（例如亮在肺部以外的區域）。
# 3) 把 Grad-CAM 目標層從 conv3 改成 conv2（更淺、解析度更高的 64x64 特徵圖），
#    熱力圖的定位變得更細還是更粗？這說明了「深層語意 vs 淺層定位」的取捨。
# ------------------------------------------------------------------
