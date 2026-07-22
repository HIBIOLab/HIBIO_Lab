"""
第 19 課：Attention 機制與 Transformer
資料集：MNIST，沿用第 18 課『把 28x28 圖片橫切成 28 個時間步』的序列設定，
        方便直接跟 LSTM/GRU 做比較。

這裡自己手刻一個最小的 self-attention（明確拆出 Query / Key / Value），
疊成一個迷你 Transformer encoder 來做分類，並且：
  1) 把 self-attention 的權重矩陣畫出來，具體看到模型『在關注哪些列』。
  2) 跟第 18 課的 LSTM/GRU 比較準確率、參數量、訓練時間。
"""

import sys
import time
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

SEQ_LEN = 28
INPUT_SIZE = 28
D_MODEL = 64


def get_loaders(batch_size=128, train_subset=6000, test_subset=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = MNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = MNIST(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False))


class SimpleSelfAttention(nn.Module):
    """明確拆出 Query / Key / Value 的單頭 self-attention，方便直接看權重。"""

    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** 0.5

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq, seq)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        return out, attn_weights


class TinyTransformerClassifier(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, input_size=INPUT_SIZE, d_model=D_MODEL, n_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.attn = SimpleSelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, n_classes)
        self.last_attn_weights = None  # 訓練/推論後可以從這裡取出權重畫圖

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, seq_len=28, input_size=28)
        x = x.squeeze(1)
        h = self.input_proj(x) + self.pos_embedding  # 加上位置編碼，因為 attention 本身不知道順序

        attn_out, attn_weights = self.attn(h)
        h = self.norm1(h + attn_out)          # 殘差連接 + LayerNorm
        h = self.norm2(h + self.ffn(h))       # 殘差連接 + LayerNorm

        self.last_attn_weights = attn_weights.detach()
        pooled = h.mean(dim=1)                # 簡單用平均池化整合整個序列的資訊
        return self.classifier(pooled)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, train_loader, test_loader, epochs=6, lr=1e-3):
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

    model = TinyTransformerClassifier()
    print(f"參數量: {count_params(model):,}\n")
    acc, elapsed = train_and_eval(model, train_loader, test_loader)
    print(f"\nTinyTransformer: test_acc={acc:.4f}  訓練耗時={elapsed:.1f}s")
    print("（可以拿這個數字跟第 18 課的 LSTM/GRU 比較：Transformer 因為")
    print(" 每個時間步可以『同時』關注所有其他時間步，天生可以平行運算，")
    print(" 不像 RNN 必須一步一步循序處理。）")

    print("\n== 畫出一個測試樣本的 self-attention 權重 ==")
    model.eval()
    sample_x, sample_y = test_loader.dataset[0]
    with torch.no_grad():
        model(sample_x.unsqueeze(0).to(DEVICE))
    attn = model.last_attn_weights[0].cpu().numpy()  # (seq_len, seq_len)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(sample_x[0], cmap="gray")
    axes[0].set_title(f"輸入影像 (數字 {sample_y})")
    axes[0].axis("off")

    im = axes[1].imshow(attn, cmap="viridis")
    axes[1].set_xlabel("Key 位置（被關注的列）")
    axes[1].set_ylabel("Query 位置（正在查詢的列）")
    axes[1].set_title("Self-Attention 權重矩陣")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "19_attention_weights.png"
    fig.savefig(out_path, dpi=120)
    print(f"圖片已存到: {out_path}")
    print("=> 矩陣中第 i 列代表『第 i 個時間步（圖片第 i 列像素）』在做分類")
    print("   判斷時，分別給了其他每個時間步多少關注權重，權重和為 1。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 19 課）：
# 1) 用自己的話解釋 Query、Key、Value 在 SimpleSelfAttention 中各自
#    對應程式碼的哪一行、扮演什麼角色。
# 2) 把 self.pos_embedding 拿掉（改成不加位置編碼），重新訓練，觀察
#    準確率是否下降？為什麼 attention 機制本身『不知道』輸入的順序？
# ------------------------------------------------------------------
