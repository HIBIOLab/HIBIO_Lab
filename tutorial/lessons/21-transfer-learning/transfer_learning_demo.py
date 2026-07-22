"""
第 21 課：遷移學習 —— Pretrain / Freeze / Fine-tune
資料集：CIFAR-10（torchvision 自動下載）+ torchvision 內建的 ImageNet 預訓練 ResNet18

重要提醒：CIFAR-10 仍然是「自然影像」（貓、狗、飛機、船…），它跟 ImageNet
的影像風格很接近，所以這裡示範的遷移效果會比「自然影像 -> 醫學影像」的
真實情境樂觀很多。這支程式的重點是讓你看懂 pretrain/freeze/fine-tune
『三種做法在程式碼上怎麼實作、行為上有什麼差異』，至於遷移到醫學影像上
效果打多少折扣，請見程式最後的說明與 index.html 第 21 課的討論。

比較三種訓練方式，在「訓練資料很少」的情境下：
  1) From scratch：完全隨機初始化，自己從頭訓練
  2) Freeze（線性探測 / linear probing）：凍結 ImageNet 預訓練權重，
     只訓練新加上去的分類頭
  3) Fine-tune：用 ImageNet 預訓練權重初始化，全部參數一起用較小的
     learning rate 繼續訓練
"""

import sys
import time
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import CIFAR10

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 96  # resize 大一點讓 ResNet18 的卷積堆疊有合理的感受野


def get_loaders(batch_size=64, train_subset=800, test_subset=1000):
    """刻意只取 800 張訓練圖，模擬醫學影像資料量有限、無法從頭訓練大模型的情境。"""
    tfm = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # ImageNet 預訓練模型看慣的輸入分布，用同樣的 mean/std 正規化
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_full = CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False))


def build_from_scratch_model(n_classes=10):
    """跟 ResNet18 架構相同，但不載入 ImageNet 權重，完全隨機初始化。"""
    return models.resnet18(weights=None, num_classes=n_classes)


def build_frozen_model(n_classes=10):
    """Freeze：載入 ImageNet 預訓練權重，把所有 backbone 參數設成不可訓練，
    只有新換上去的最後一層（分類頭）會被訓練 —— 這叫『linear probing』。"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # 凍結整個 backbone
    model.fc = nn.Linear(model.fc.in_features, n_classes)  # 新的分類頭，預設 requires_grad=True
    return model


def build_finetune_model(n_classes=10):
    """Fine-tune：一樣載入 ImageNet 預訓練權重當作初始值，但『不』凍結，
    全部參數都會被更新，只是通常搭配比從頭訓練更小的 learning rate，
    避免破壞預訓練學到的通用特徵。"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def train_and_eval(model, train_loader, test_loader, epochs, lr):
    model.to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
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
        print(f"    epoch {epoch+1}/{epochs}  test_acc={acc:.4f}")
    return acc, time.time() - start


def main():
    print(f"使用裝置: {DEVICE}")
    train_loader, test_loader = get_loaders()
    print(f"訓練集: {len(train_loader.dataset)} 張（刻意調小，模擬資料有限的情境）"
          f"，測試集: {len(test_loader.dataset)} 張\n")

    results = {}

    print("== (1) From scratch：完全隨機初始化 ==")
    model_scratch = build_from_scratch_model()
    print(f"  可訓練參數: {count_trainable_params(model_scratch):,} / "
          f"總參數: {count_total_params(model_scratch):,}")
    acc, t = train_and_eval(model_scratch, train_loader, test_loader, epochs=8, lr=1e-3)
    results["From scratch"] = (acc, t, count_trainable_params(model_scratch))

    print("\n== (2) Freeze：凍結 ImageNet 預訓練權重，只訓練新的分類頭 ==")
    model_frozen = build_frozen_model()
    print(f"  可訓練參數: {count_trainable_params(model_frozen):,} / "
          f"總參數: {count_total_params(model_frozen):,}")
    acc, t = train_and_eval(model_frozen, train_loader, test_loader, epochs=8, lr=1e-3)
    results["Freeze (linear probe)"] = (acc, t, count_trainable_params(model_frozen))

    print("\n== (3) Fine-tune：用 ImageNet 權重初始化，全部參數一起用小 LR 微調 ==")
    model_finetune = build_finetune_model()
    print(f"  可訓練參數: {count_trainable_params(model_finetune):,} / "
          f"總參數: {count_total_params(model_finetune):,}")
    acc, t = train_and_eval(model_finetune, train_loader, test_loader, epochs=8, lr=1e-4)
    results["Fine-tune"] = (acc, t, count_trainable_params(model_finetune))

    print("\n== 總結比較（訓練資料只有 800 張，訓練 8 epoch）==")
    print(f"{'做法':28s} {'test_acc':>10s} {'可訓練參數':>14s} {'耗時':>8s}")
    for name, (acc, t, n_params) in results.items():
        print(f"{name:28s} {acc:10.4f} {n_params:14,} {t:7.1f}s")

    print("\n=> 在資料量有限的情況下，直接從頭訓練 ResNet18 這種較大的模型通常")
    print("   表現最差（沒有足夠資料學會所有參數）；Freeze 只需要訓練一個線性")
    print("   分類頭，參數量少很多、訓練最快，通常也比從頭訓練好；Fine-tune")
    print("   讓預訓練特徵能針對新任務微調，往往能拿到最好的準確率，但代價是")
    print("   要訓練的參數量最多、也最容易在資料極少時 overfit，所以通常要搭配")
    print("   較小的 learning rate。")

    print("\n" + "=" * 70)
    print("重要提醒：這裡的『下游任務』是 CIFAR-10，跟 ImageNet 一樣都是自然")
    print("影像（貓、狗、船…），domain gap 很小，遷移效果會比較樂觀。真正把")
    print("ImageNet 預訓練模型搬到醫學影像（灰階 CT/X光、超音波、病理切片）")
    print("時，兩者的影像統計特性差異很大：色彩分布、紋理、對比機制都不同，")
    print("ImageNet 學到的『貓耳朵』『車輪』這類高階語意特徵幾乎用不上。")
    print("研究（如 Raghu et al., 2019, Transfusion）發現，在許多醫學影像任務")
    print("上，ImageNet 預訓練帶來的好處主要來自『更好的權重初始值、訓練收斂")
    print("更快更穩定』，而不是真的『看得懂病灶』；但即使如此，經驗上預訓練")
    print("在醫學影像資料量有限時，通常還是比從頭訓練更快收斂、效果更穩定，")
    print("所以『不是完全無痛的遷移，但依然值得做』，是業界與文獻中普遍的共識。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 21 課）：
# 1) 把 train_subset 從 800 調大到 5000，重新比較三種做法的差距是否
#    縮小？這說明了「資料量越大，pretrain 帶來的優勢越不明顯」。
# 2) 試著實作「漸進解凍 (progressive unfreezing)」：先用 build_frozen_model
#    訓練幾個 epoch，再把最後一個 block（model.layer4）的 requires_grad
#    打開，用更小的 learning rate 繼續訓練，比較跟純 freeze / 純
#    fine-tune 的差異。
# 3) 為什麼 fine-tune 通常要用比 from-scratch 更小的 learning rate？
#    如果用一樣大的 learning rate 做 fine-tune 可能會發生什麼問題？
# ------------------------------------------------------------------
