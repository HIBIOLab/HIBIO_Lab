"""
第 22 課：醫學影像常用的 Data Augmentation 技術
資料集：
  (A) pydicom 內建的真實 CT 範例（沿用第 8 課），示範各種增強技巧的視覺效果
  (B) MNIST，用極少量訓練資料（300 張）驗證「有無 augmentation」對泛化能力的實際影響

這一課想傳達的重點：medical imaging 的 augmentation 不能照抄一般電腦視覺的
套路（隨便轉、隨便翻），必須考慮解剖學上的合理性，否則等於是在教模型
學習『不存在的病人』。
"""

import sys
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch
import torch.nn as nn
from pydicom.data import get_testdata_file
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import functional as VF

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# Part A：在一張真實 CT 影像上展示各種常見的醫學影像 augmentation
# ------------------------------------------------------------------

def load_ct_image():
    ds = pydicom.dcmread(get_testdata_file("CT_small.dcm"))
    img = ds.pixel_array.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())  # normalize 到 0-1
    return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


def small_rotation(img, degrees=10):
    """醫學影像通常只能做小角度旋轉（模擬病人擺位的些微差異），
    不能像一般物件辨識那樣隨意轉 90°/180°，否則會違反解剖學方位。"""
    return VF.rotate(img.unsqueeze(0), degrees)[0]


def gamma_correction(img, gamma=1.6):
    """模擬不同機台 / 曝光條件下，影像亮度對比的差異。"""
    return torch.clamp(img, 0, 1) ** gamma


def add_gaussian_noise(img, std=0.05):
    """模擬感測器雜訊 / 低劑量掃描下的雜訊增加。"""
    return torch.clamp(img + torch.randn_like(img) * std, 0, 1)


def elastic_deform(img, alpha=30.0, sigma=5.0):
    """彈性形變：模擬組織的些微變形，是醫學影像分割任務中最常見、
    效果也最好的增強方式之一（U-Net 原始論文就是用這招）。
    強度(alpha)不能太大，否則會扭曲出解剖學上不合理的形狀。"""
    transform = transforms.ElasticTransform(alpha=alpha, sigma=sigma)
    return transform(img.unsqueeze(0))[0]


def random_erasing(img, scale=(0.02, 0.08)):
    """隨機遮蓋一小塊區域，模擬影像中的偽影(artifact)或部分遮擋，
    強迫模型不要只靠影像中的某一小塊區域做判斷。"""
    eraser = transforms.RandomErasing(p=1.0, scale=scale, value=0.0)
    return eraser(img.unsqueeze(0))[0]


def unsafe_horizontal_flip(img):
    """⚠️ 示範一個『不一定安全』的增強：水平翻轉。
    對於左右不對稱的解剖構造（例如胸腔中心臟偏左），隨意水平翻轉
    可能會產生解剖學上不存在的影像，套用前務必先確認任務是否適合。"""
    return VF.hflip(img)


def visualize_augmentations():
    img = load_ct_image()

    augmentations = {
        "原圖": img,
        "小角度旋轉 (10°)": small_rotation(img),
        "Gamma 校正 (亮度/對比)": gamma_correction(img),
        "高斯雜訊": add_gaussian_noise(img),
        "彈性形變 (Elastic)": elastic_deform(img),
        "隨機遮蓋 (Erasing)": random_erasing(img),
        "[謹慎] 水平翻轉": unsafe_horizontal_flip(img),
    }

    fig, axes = plt.subplots(1, len(augmentations), figsize=(3 * len(augmentations), 3.4))
    for ax, (name, aug_img) in zip(axes, augmentations.items()):
        ax.imshow(aug_img.squeeze(0).numpy(), cmap="gray")
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_path = OUTPUT_DIR / "22_augmentation_examples.png"
    fig.savefig(out_path, dpi=120)
    print(f"圖片已存到: {out_path}")


# ------------------------------------------------------------------
# Part B：用極少量訓練資料，量化 augmentation 對泛化能力的實際幫助
# ------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_mnist(n_train=300, n_test=2000):
    train_full = MNIST(root=str(DATA_DIR), train=True, download=True)
    test_full = MNIST(root=str(DATA_DIR), train=False, download=True)

    X_train = train_full.data[:n_train].float() / 255.0
    y_train = train_full.targets[:n_train]
    X_test = test_full.data[:n_test].float() / 255.0
    y_test = test_full.targets[:n_test]
    return X_train.unsqueeze(1), y_train, X_test.unsqueeze(1), y_test


def train_with_augmentation(X_train, y_train, X_test, y_test, use_augmentation, epochs=40):
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_test_dev, y_test_dev = X_test.to(DEVICE), y_test.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 32):
            idx = perm[i:i + 32]
            xb, yb = X_train[idx].to(DEVICE), y_train[idx].to(DEVICE)

            if use_augmentation:
                # 每個 batch 都用隨機小角度旋轉 + 輕微雜訊，等於每次看到的都是「新」的影像
                angle = float(torch.empty(1).uniform_(-12, 12))
                xb = VF.rotate(xb, angle)
                xb = torch.clamp(xb + torch.randn_like(xb) * 0.05, 0, 1)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train.to(DEVICE)).argmax(1) == y_train.to(DEVICE)).float().mean().item()
        test_acc = (model(X_test_dev).argmax(1) == y_test_dev).float().mean().item()
    return train_acc, test_acc


def quantify_augmentation_effect():
    print(f"使用裝置: {DEVICE}")
    X_train, y_train, X_test, y_test = get_mnist(n_train=300)
    print(f"刻意只用 {len(X_train)} 張訓練資料，模擬醫學影像資料量有限的情境\n")

    print("== 沒有 augmentation ==")
    train_acc_no, test_acc_no = train_with_augmentation(X_train, y_train, X_test, y_test, use_augmentation=False)
    print(f"train_acc={train_acc_no:.4f}  test_acc={test_acc_no:.4f}  gap={train_acc_no-test_acc_no:.4f}")

    print("\n== 有 augmentation（隨機旋轉 ±12° + 輕微雜訊）==")
    train_acc_yes, test_acc_yes = train_with_augmentation(X_train, y_train, X_test, y_test, use_augmentation=True)
    print(f"train_acc={train_acc_yes:.4f}  test_acc={test_acc_yes:.4f}  gap={train_acc_yes-test_acc_yes:.4f}")

    print(f"\n總結：test accuracy {'提升' if test_acc_yes > test_acc_no else '沒有提升'} "
          f"{abs(test_acc_yes-test_acc_no):.4f}")
    print("=> 在資料量非常有限時，augmentation 等於用同一批影像『生出更多變化版本』")
    print("   讓模型看，訓練資料實際上的多樣性增加了，train/test 差距通常也會縮小。")


def main():
    print("== Part A：在真實 CT 影像上展示常見的醫學影像 augmentation ==")
    visualize_augmentations()

    print("\n== Part B：用極少訓練資料驗證 augmentation 對泛化能力的幫助 ==")
    quantify_augmentation_effect()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 22 課）：
# 1) 在 visualize_augmentations() 中把 small_rotation 的角度從 10 改成
#    90，觀察對這張 CT 影像來說，這個角度還合理嗎？為什麼醫學影像通常
#    只能用「小角度」旋轉？
# 2) 在 quantify_augmentation_effect() 中把 n_train 從 300 調大到 3000，
#    重新比較有無 augmentation 的差距是否縮小？這跟第 16 課「資料越多
#    越不容易 overfit」的結論是否一致？
# 3) 針對「胸部 X 光肺炎分類」任務，從本課展示的 6 種增強中選出你認為
#    合適的 3 種、不合適的 1 種，並說明理由。
# ------------------------------------------------------------------
