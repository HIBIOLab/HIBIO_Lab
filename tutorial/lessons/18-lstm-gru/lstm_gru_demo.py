"""
第 18 課：RNN 家族 —— LSTM 與 GRU
資料集：S&P 500 Stocks（25 年每日股價）
        放在 tutorial/data/SP500/（sp500_stocks.csv、sp500_companies.csv）
        來源：https://www.kaggle.com/datasets/darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily

序列預測任務：拿某一檔股票（預設 AAPL）過去 SEQ_LEN 天的收盤價，預測「下一天」
的收盤價。這是 RNN 最典型的應用場景 —— 時間序列。我們在同一個任務上比較
LSTM 與 GRU 的：測試誤差（RMSE / MAE / MAPE）、參數量、訓練時間。

備註：股價預測本身極難、雜訊很大，這裡的重點不是「準到能拿去交易」，而是
用一個真實的時間序列，把 LSTM 與 GRU 的行為差異講清楚。
"""

import sys
import time
from pathlib import Path

if sys.stdout.encoding is None or "utf" not in sys.stdout.encoding.lower():
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SP500_CSV = DATA_DIR / "SP500" / "sp500_stocks.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TICKER = "AAPL"     # 想換別檔股票，改這裡（例如 "MSFT"、"NVDA"）
SEQ_LEN = 60        # 用過去 60 個交易日
HIDDEN_SIZE = 64
TEST_DAYS = 252     # 最後約一年的交易日當測試集


def load_sequences():
    """讀 CSV、取出 TICKER 的收盤價、縮放、切成 (過去 SEQ_LEN 天 -> 下一天) 的序列。"""
    if not SP500_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {SP500_CSV}。請確認 SP500 資料集已放在 {SP500_CSV.parent}。")

    print(f"讀取 {SP500_CSV.name}（約 275MB，請稍候）...")
    df = pd.read_csv(SP500_CSV, usecols=["date", "close", "symbol"], parse_dates=["date"])
    one = df[df["symbol"] == TICKER].dropna(subset=["close"]).sort_values("date")
    prices = one["close"].values.reshape(-1, 1).astype(np.float32)
    dates = one["date"].values
    print(f"{TICKER}: 共 {len(prices)} 個交易日（{one['date'].min().date()} ~ {one['date'].max().date()}）")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X, y = [], []
    for i in range(SEQ_LEN, len(scaled)):
        X.append(scaled[i - SEQ_LEN:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X, dtype=np.float32).reshape(-1, SEQ_LEN, 1)  # (N, seq, 1 feature)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    split = len(X) - TEST_DAYS
    data = {
        "X_train": torch.from_numpy(X[:split]), "y_train": torch.from_numpy(y[:split]),
        "X_test": torch.from_numpy(X[split:]), "y_test": torch.from_numpy(y[split:]),
    }
    test_dates = dates[SEQ_LEN + split:]
    return data, scaler, test_dates


class RNNRegressor(nn.Module):
    def __init__(self, cell_type="lstm"):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.rnn = rnn_cls(input_size=1, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        out, _ = self.rnn(x)        # out: (batch, seq_len, hidden)
        last_step = out[:, -1, :]   # 只取最後一個時間步的輸出來預測
        return self.fc(last_step)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_eval(model, data, epochs=25, lr=1e-3, batch_size=32):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["y_train"]), batch_size=batch_size, shuffle=True)
    X_test, y_test = data["X_test"].to(DEVICE), data["y_test"].to(DEVICE)

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
        with torch.no_grad():
            val_loss = loss_fn(model(X_test), y_test).item()
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:2d}/{epochs}  test_mse(scaled)={val_loss:.5f}")
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test).cpu().numpy()
    return pred_scaled, elapsed


def dollar_metrics(pred_scaled, y_test, scaler):
    y_pred = scaler.inverse_transform(pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_test.numpy()).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return y_pred, y_true, rmse, mae, mape


def main():
    print(f"使用裝置: {DEVICE}")
    data, scaler, test_dates = load_sequences()
    print(f"用過去 {SEQ_LEN} 天預測下一天；訓練 {len(data['X_train'])} 段、"
          f"測試 {len(data['X_test'])} 段\n")

    print("== LSTM ==")
    torch.manual_seed(0)
    lstm = RNNRegressor("lstm")
    print(f"參數量: {count_params(lstm):,}")
    lstm_pred, lstm_time = train_and_eval(lstm, data)
    y_pred_l, y_true, rmse_l, mae_l, mape_l = dollar_metrics(lstm_pred, data["y_test"], scaler)

    print("\n== GRU ==")
    torch.manual_seed(0)
    gru = RNNRegressor("gru")
    print(f"參數量: {count_params(gru):,}")
    gru_pred, gru_time = train_and_eval(gru, data)
    y_pred_g, _, rmse_g, mae_g, mape_g = dollar_metrics(gru_pred, data["y_test"], scaler)

    print(f"\n最終比較（{TICKER} 未來 {TEST_DAYS} 個交易日的預測誤差）：")
    print(f"  LSTM: RMSE=${rmse_l:.2f}  MAE=${mae_l:.2f}  MAPE={mape_l:.2f}%  "
          f"參數={count_params(lstm):,}  訓練耗時={lstm_time:.1f}s")
    print(f"  GRU : RMSE=${rmse_g:.2f}  MAE=${mae_g:.2f}  MAPE={mape_g:.2f}%  "
          f"參數={count_params(gru):,}  訓練耗時={gru_time:.1f}s")
    print("\n=> LSTM 有 3 個 gate（input/forget/output）+ cell state，GRU 只有 2 個")
    print("   gate（reset/update），少了一組參數，通常訓練更快、參數更少，準確率")
    print("   則常常與 LSTM 相差無幾——這也是為什麼資料量有限或需要快速迭代時，")
    print("   GRU 常是優先考慮的選項。")

    # 視覺化：實際 vs 兩個模型的預測
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test_dates, y_true, color="black", linewidth=1.5, label="實際收盤價")
    ax.plot(test_dates, y_pred_l, color="#F44336", linewidth=1.1, linestyle="--",
            label=f"LSTM 預測 (MAPE {mape_l:.1f}%)")
    ax.plot(test_dates, y_pred_g, color="#2196F3", linewidth=1.1, linestyle="--",
            label=f"GRU 預測 (MAPE {mape_g:.1f}%)")
    ax.set_title(f"{TICKER}：LSTM / GRU 對測試期收盤價的預測")
    ax.set_ylabel("價格 ($)")
    ax.legend()
    fig.tight_layout()
    out_path = OUTPUT_DIR / "18_lstm_gru_sp500.png"
    fig.savefig(out_path, dpi=120)
    print(f"\n圖片已存到: {out_path}")


if __name__ == "__main__":
    main()

# ------------------------------------------------------------------
# 課後練習（對照 index.html 第 18 課）：
# 1) 說明 LSTM 的 forget gate 作用，為什麼它有助於解決長期依賴問題？
#    可以到 PyTorch 文件查 nn.LSTM 的公式，找出 forget gate 對應的項。
# 2) 把 HIDDEN_SIZE 從 64 降到 16，重新比較 LSTM/GRU 的參數量與預測誤差
#    差距是否隨著模型變小而縮小或放大？
# 3) 把 TICKER 換成另一檔股票（例如波動更大的 "NVDA"），MAPE 是否變高？
#    想想看：這跟「越難預測的序列、模型誤差越大」是否一致？
# ------------------------------------------------------------------
