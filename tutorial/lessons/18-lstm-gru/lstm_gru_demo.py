"""
第 18 課：RNN 家族 —— LSTM 與 GRU
資料集：MNIST，但這裡刻意把每張 28x28 的圖『橫著切成 28 個時間步，每步
        28 個像素值』，當作一個序列分類任務餵給 RNN（這是教學上常見的
        RNN demo 用法：把影像視為序列來練習 LSTM/GRU，並不是說 RNN
        比 CNN 更適合處理影像 —— 這幾課的重點是比較模型本身的行為）。

比較 LSTM 與 GRU 在同樣任務上的準確率、參數量、與訓練速度。
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
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 28    # 把圖片的每一列當作一個時間步
INPUT_SIZE = 28  # 每個時間步是一列的 28 個像素值
HIDDEN_SIZE = 64


def get_loaders(batch_size=128, train_subset=6000, test_subset=2000):
    tfm = transforms.Compose([transforms.ToTensor()])
    train_full = MNIST(root=str(DATA_DIR), train=True, download=True, transform=tfm)
    test_full = MNIST(root=str(DATA_DIR), train=False, download=True, transform=tfm)
    train_set = torch.utils.data.Subset(train_full, range(train_subset))
    test_set = torch.utils.data.Subset(test_full, range(test_subset))
    return (DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(test_set, batch_size=batch_size, shuffle=False))


class RNNClassifier(nn.Module):
    def __init__(self, cell_type="lstm"):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.rnn = rnn_cls(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 10)

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, 28, 28)，把每一列視為一個時間步
        x = x.squeeze(1)
        out, _ = self.rnn(x)          # out: (batch, seq_len, hidden)
        last_step = out[:, -1, :]     # 只取最後一個時間步的輸出來分類
        return self.fc(last_step)


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
    elapsed = time.time() - start
    return acc, elapsed


def main():
    print(f"使用裝置: {DEVICE}")
    print(f"把每張 28x28 的圖當作長度 {SEQ_LEN}、每步 {INPUT_SIZE} 維的序列\n")
    train_loader, test_loader = get_loaders()

    print("== LSTM ==")
    lstm = RNNClassifier("lstm")
    print(f"參數量: {count_params(lstm):,}")
    lstm_acc, lstm_time = train_and_eval(lstm, train_loader, test_loader)

    print("\n== GRU ==")
    gru = RNNClassifier("gru")
    print(f"參數量: {count_params(gru):,}")
    gru_acc, gru_time = train_and_eval(gru, train_loader, test_loader)

    print(f"\n最終比較：")
    print(f"  LSTM: test_acc={lstm_acc:.4f}  參數={count_params(lstm):,}  訓練耗時={lstm_time:.1f}s")
    print(f"  GRU : test_acc={gru_acc:.4f}  參數={count_params(gru):,}  訓練耗時={gru_time:.1f}s")
    print("\n=> LSTM 有 3 個 gate（input/forget/output）+ cell state，")
    print("   GRU 只有 2 個 gate（reset/update），少了一組參數，通常訓練")
    print("   更快、參數更少，準確率則常常與 LSTM 相差無幾——這也是為什麼")
    print("   在資料量有限或需要快速迭代時，GRU 常是優先考慮的選項。")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 18 課）：
# 1) 說明 LSTM 的 forget gate 作用，為什麼它有助於解決長期依賴問題？
#    可以到 PyTorch 文件查 nn.LSTM 的公式，找出 forget gate 對應的項。
# 2) 把 HIDDEN_SIZE 從 64 降到 16，重新比較 LSTM/GRU 的參數量與準確率
#    差距是否隨著模型變小而縮小或放大？
# ------------------------------------------------------------------
