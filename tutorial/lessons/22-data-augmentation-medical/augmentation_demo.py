"""
第 22 課：醫學影像常用的 Data Augmentation 技術
資料集：COVID-19 Chest X-ray（胸部 X 光三分類：Covid / Normal / Viral Pneumonia）
        放在 tutorial/data/Covid19-classification/（train/ 與 test/，各含三個類別資料夾）
        來源：https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

兩個部分：
  (A) 在一張真實胸部 X 光上，視覺化各種常見的醫學影像 augmentation，並點出
      哪些手法「合理」、哪些要「謹慎」（例如水平翻轉會違反『心臟偏左』的
      解剖事實）。
  (B) 這個資料集的訓練資料只有約 250 張，正是「醫學影像資料量有限」的真實
      情境。我們用同一個小 CNN，比較「有無 augmentation」對三分類泛化能力
      的實際影響。

重點：medical imaging 的 augmentation 不能照抄一般電腦視覺的套路（隨便轉、
隨便翻），必須考慮解剖學上的合理性，否則等於在教模型學習『不存在的病人』。
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
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as VF

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COVID_DIR = DATA_DIR / "Covid19-classification"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = ["Covid", "Normal", "Viral Pneumonia"]  # 對應 0, 1, 2
IMG_SIZE = 128        # Part B 訓練用的影像大小
VIZ_SIZE = 256        # Part A 視覺化用的影像大小（大一點看得清楚）


def load_xray(path, size):
    """讀一張胸部 X 光，轉灰階、縮放、normalize 到 0-1，回傳 (1, H, W) tensor。"""
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ------------------------------------------------------------------
# Part A：在一張真實胸部 X 光上展示各種常見的醫學影像 augmentation
# ------------------------------------------------------------------

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
    胸部 X 光的心臟通常偏左，隨意水平翻轉會產生解剖學上不存在的影像
    （心臟跑到右邊 = 罕見的『右位心』），可能教模型學到錯誤的空間先驗。"""
    return VF.hflip(img)


def visualize_augmentations():
    # 挑一張 Covid 類別的胸部 X 光來示範
    covid_imgs = sorted((COVID_DIR / "train" / "Covid").glob("*"))
    img = load_xray(covid_imgs[0], VIZ_SIZE)

    augmentations = {
        "原圖 (Covid X 光)": img,
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
# Part B：用這個資料量有限的資料集，量化 augmentation 對泛化能力的幫助
# ------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 128 -> 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.AdaptiveAvgPool2d(4),                                      # -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_covid_split(split):
    """讀取 Covid19-classification/<split> 底下三個類別的所有影像。"""
    split_dir = COVID_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"找不到 {split_dir}。請確認 COVID-19 影像資料集已放在 {COVID_DIR}（含 train/ 與 test/）。")

    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((split_dir / cls).glob("*")):
            xs.append(load_xray(p, IMG_SIZE))
            ys.append(label)
    X = torch.stack(xs)                       # (N, 1, 128, 128)
    y = torch.tensor(ys, dtype=torch.long)
    return X, y


def safe_augment(xb):
    """對一個 batch 套用『解剖學上安全』的隨機增強：小角度旋轉 + 亮度擾動 + 輕微雜訊。
    刻意不做水平翻轉（見 Part A 的說明）。"""
    angle = float(torch.empty(1).uniform_(-10, 10))
    xb = VF.rotate(xb, angle)
    gamma = float(torch.empty(1).uniform_(0.7, 1.4))
    xb = torch.clamp(xb, 0, 1) ** gamma
    xb = torch.clamp(xb + torch.randn_like(xb) * 0.03, 0, 1)
    return xb


def train_covid_cnn(X_train, y_train, X_test, y_test, use_augmentation, epochs=30, seed=0):
    torch.manual_seed(seed)
    model = SmallCNN(n_classes=len(CLASSES)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    X_test_dev, y_test_dev = X_test.to(DEVICE), y_test.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 16):
            idx = perm[i:i + 16]
            xb, yb = X_train[idx].to(DEVICE), y_train[idx].to(DEVICE)
            if use_augmentation:
                xb = safe_augment(xb)
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
    X_train, y_train = load_covid_split("train")
    X_test, y_test = load_covid_split("test")
    counts = {cls: int((y_train == i).sum()) for i, cls in enumerate(CLASSES)}
    print(f"訓練資料 {len(X_train)} 張（{counts}）、測試資料 {len(X_test)} 張——資料量非常有限\n")

    print("== 沒有 augmentation ==")
    tr_no, te_no = train_covid_cnn(X_train, y_train, X_test, y_test, use_augmentation=False)
    print(f"train_acc={tr_no:.4f}  test_acc={te_no:.4f}  gap={tr_no-te_no:.4f}")

    print("\n== 有 augmentation（安全增強：旋轉 ±10° + 亮度擾動 + 輕微雜訊）==")
    tr_yes, te_yes = train_covid_cnn(X_train, y_train, X_test, y_test, use_augmentation=True)
    print(f"train_acc={tr_yes:.4f}  test_acc={te_yes:.4f}  gap={tr_yes-te_yes:.4f}")

    print(f"\n總結：test accuracy {'提升' if te_yes > te_no else '沒有提升'} "
          f"{abs(te_yes-te_no):.4f}；train/test 差距 "
          f"{'縮小' if (tr_yes-te_yes) < (tr_no-te_no) else '沒有縮小'}")
    print("=> 資料量非常有限時，augmentation 等於用同一批 X 光『生出更多合理的變化版本』")
    print("   讓模型看，訓練資料的有效多樣性增加，通常能提升 test accuracy、縮小過擬合。")

    # 長條圖：有無 augmentation 的 train / test accuracy
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(2)
    ax.bar(x - 0.2, [tr_no, tr_yes], width=0.4, label="train accuracy")
    ax.bar(x + 0.2, [te_no, te_yes], width=0.4, label="test accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(["沒有 augmentation", "有 augmentation"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("COVID-19 胸部 X 光三分類：augmentation 對泛化的影響")
    ax.legend()
    for xi, (tr, te) in zip(x, [(tr_no, te_no), (tr_yes, te_yes)]):
        ax.text(xi - 0.2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=9)
        ax.text(xi + 0.2, te + 0.01, f"{te:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "22_augmentation_covid_effect.png"
    fig.savefig(out_path, dpi=120)
    print(f"圖片已存到: {out_path}")


def main():
    print("== Part A：在真實胸部 X 光上展示常見的醫學影像 augmentation ==")
    visualize_augmentations()

    print("\n== Part B：用資料量有限的 COVID X 光驗證 augmentation 對泛化能力的幫助 ==")
    quantify_augmentation_effect()


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 22 課）：
# 1) 在 visualize_augmentations() 中把 small_rotation 的角度從 10 改成
#    90，觀察對這張胸部 X 光來說這個角度還合理嗎？為什麼醫學影像通常
#    只能用「小角度」旋轉？
# 2) 把 safe_augment() 裡加入 unsafe_horizontal_flip（水平翻轉），重新跑
#    Part B，test accuracy 有沒有變差？結合心臟偏左的解剖事實想想為什麼。
# 3) 針對「胸部 X 光三分類」任務，從 Part A 展示的 6 種增強中選出你認為
#    合適的 3 種、不合適的 1 種，並說明理由。
# ------------------------------------------------------------------
