# 醫學影像 AI 教學區 — 程式碼

這個資料夾對應 [./tutorial/index.html](./index.html) 的 22 堂課，每堂課都有一個獨立資料夾（`lessons/01-holdout/` ~ `lessons/22-data-augmentation-medical/`），內含**可直接執行**的訓練 / 示範程式碼。

## 環境設定（conda 建環境 + pip 裝 GPU 版 PyTorch）

用 conda 建立乾淨獨立的 Python 環境，再用 `pip` 安裝套件（包含 GPU 版 PyTorch）——這是目前最穩定、最不容易卡住的組合：conda 只負責「環境隔離」，pip 直接從 PyTorch 官方 wheel 索引裝最新的 GPU 版本，不會遇到 conda 解析 `pytorch-cuda` 套件時可能很慢的問題。

```bash
cd tutorial
conda create -n hibio-tutorial python=3.10 -y
conda activate hibio-tutorial
pip install -r requirements.txt
```

`requirements.txt` 最後三行已經指定 `--extra-index-url https://download.pytorch.org/whl/cu126`，`pip install` 會自動抓對應 **CUDA 12.6** 的 GPU 版 PyTorch（沒有 NVIDIA 顯示卡也能裝，會自動退回 CPU 版，只是跑起來比較慢）。這個組合已經實際測試過：`torch.cuda.is_available()` 會回傳 `True`，且能正確辨識到顯卡。

有 NVIDIA GPU 的話 PyTorch 會自動使用 CUDA 加速；沒有 GPU 也能跑，只是深度學習那幾課（16-21 課）會慢一些。

## Windows GPU 環境設定（第一次裝 CUDA 的人看這裡）

如果你是 Windows 第一次要跑 GPU 深度學習環境，看到 CUDA / cuDNN / CUDA Toolkit 這幾個名詞會搞混很正常，這裡先講清楚它們的關係，再給步驟。

### 先搞懂：這幾個東西差在哪

- **NVIDIA 顯示卡驅動 (Driver)**：讓作業系統認得你的顯卡，`nvidia-smi` 這個指令就是驅動裝好才會有的工具。**這個一定要裝，conda/pip 都沒辦法幫你裝，要自己去 NVIDIA 官網下載安裝。**
- **CUDA Toolkit**：NVIDIA 提供給「開發者編譯 CUDA 程式」用的完整工具包（含編譯器 `nvcc`、函式庫、範例）。
- **cuDNN**：專門給深度學習用的加速函式庫，通常疊在 CUDA Toolkit 上面用。

### ⚡ 快速版：conda + pip（適合大多數人，也是本專案建議的方式）

**好消息：你不需要自己裝 CUDA Toolkit 或 cuDNN。** `pip install torch` 裝下來的 PyTorch wheel 已經把對應版本的 CUDA / cuDNN 執行期函式庫直接包在裡面了，只需要「顯示卡驅動」認得你的顯卡即可。步驟：

1. 確認你有 NVIDIA 顯示卡：桌面右鍵 →「NVIDIA 控制面板」→「系統資訊」，或工作管理員「效能」分頁看有沒有 GPU 0/1 顯示 NVIDIA 型號。
2. 到 [NVIDIA 官方驅動下載頁](https://www.nvidia.com/Download/index.aspx) 選你的顯卡型號，下載安裝最新驅動，安裝完**重新開機**。
3. 還沒裝過 conda 的話，先安裝 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（安裝時勾選「Add Miniconda3 to PATH」較方便，或改用內建的「Anaconda Prompt」）。
4. 開 PowerShell / cmd / Anaconda Prompt，執行：
   ```
   nvidia-smi
   ```
   如果看到顯卡型號、記憶體用量、右上角有一個 `CUDA Version: 12.x` 的欄位，代表驅動裝好了（這個版本號是驅動最高支援到的 CUDA 版本，不是你已經裝了 CUDA Toolkit）。
5. 照本頁最上面「環境設定」的步驟建立 conda 環境並用 pip 安裝：
   ```
   cd tutorial
   conda create -n hibio-tutorial python=3.10 -y
   conda activate hibio-tutorial
   pip install -r requirements.txt
   ```
6. 驗證 GPU 真的可以被 PyTorch 用到：
   ```
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   印出 `True` 和你的顯卡名稱就大功告成，**可以跳過下面「完整版」**，不需要再裝任何東西。

### 🔧 完整版：手動安裝 CUDA Toolkit + cuDNN（進階／其他用途才需要）

只有當你之後要用到「需要自己編譯 CUDA 原始碼」的套件、或要跑非 PyTorch 的框架時才需要這一節，跟上面的 conda 環境彼此獨立、互不影響：

1. **確認顯卡支援 CUDA**：到 [CUDA GPUs 支援清單](https://developer.nvidia.com/cuda-gpus) 查詢你的顯卡型號。
2. **安裝顯示卡驅動**：同上「快速版」步驟 1-2。
3. **安裝 CUDA Toolkit**：到 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) 下載跟本專案對應的版本（本專案 PyTorch 用的是 **CUDA 12.6**，建議下載 CUDA Toolkit 12.6）。建議選「exe (network)」線上安裝版，它會自動偵測系統並下載相容元件，用預設選項安裝即可。
4. **安裝 cuDNN**：到 [NVIDIA cuDNN 下載頁](https://developer.nvidia.com/cudnn) 下載（需要免費註冊 NVIDIA 帳號），下載對應 CUDA 版本的 cuDNN，解壓縮後把裡面的檔案複製到 CUDA Toolkit 安裝目錄（預設在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\`）：
   - `bin\*.dll` → CUDA 目錄下的 `bin`
   - `include\cudnn*.h` → CUDA 目錄下的 `include`
   - `lib\x64\*.lib` → CUDA 目錄下的 `lib\x64`
5. **確認環境變數**：CUDA Toolkit 安裝程式通常會自動把 `...\CUDA\v12.6\bin` 加進系統 `Path`；如果下一步 `nvcc` 指令找不到，就到「編輯系統環境變數」手動確認 / 補上這個路徑。
6. **驗證安裝**：
   ```
   nvcc --version        # 應該印出 CUDA 編譯器版本，例如 release 12.6
   nvidia-smi             # 顯示驅動與顯卡資訊
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```

### 版本對應與常見問題

- CUDA Toolkit / cuDNN 的版本要跟 PyTorch wheel 對應的 CUDA 版本一致（本專案是 **cu126**，也就是 CUDA 12.6）；版本對不上是最常見的踩雷點。但如前面提到，只是要跑這個教學區的話根本不需要裝這兩個，可以完全跳過這節。
- 顯示卡驅動要新到能支援對應的 CUDA 版本（`nvidia-smi` 右上角的 `CUDA Version` 就是驅動能支援的上限，只要它 ≥ 12.6 就沒問題）。
- `torch.cuda.is_available()` 回傳 `False` 時的排查順序：先確認 `nvidia-smi` 能不能正常顯示顯卡 → 不能就是驅動沒裝好或忘記重開機；能顯示但 torch 仍是 `False` → 通常是裝到 CPU-only 版本的 torch，確認 `pip install` 時有沒有帶到 `requirements.txt` 裡的 `--extra-index-url .../cu126`（在 `hibio-tutorial` 環境裡執行 `pip uninstall torch torchvision -y` 再 `pip install -r requirements.txt` 重裝一次通常能解決）。
- 沒有 NVIDIA 顯示卡（筆電內顯、AMD 顯卡、Apple Silicon）：直接跳過這整節，用 CPU 執行即可，教學區大部分程式都能在 CPU 上跑，只是第 16-21 課的深度學習部分會明顯變慢。

## 資料集說明

為了讓每堂課「打開就能跑」，程式碼一律使用**免帳號、可自動下載或套件內建**的資料集，不需要額外申請 Kaggle API 金鑰：

| 資料集 | 用在哪些課 | 說明 |
|---|---|---|
| MNIST（手寫數字） | 3, 4, 5, 6, 10, 16, 17, 18, 19, 22 | `torchvision.datasets.MNIST` 自動下載，約 10MB |
| CIFAR-10（10 類小型彩色圖） | 20, 21 | `torchvision.datasets.CIFAR10` 自動下載，約 170MB（University of Toronto 官方主機，下載速度視網路狀況可能較慢） |
| ImageNet 預訓練 ResNet18 權重 | 21 | `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` 自動下載，約 45MB |
| Breast Cancer Wisconsin (Diagnostic) | 1, 2, 9, 11, 12, 14 | `sklearn.datasets.load_breast_cancer`，套件內建、免下載。與 Kaggle 上同名的乳癌診斷資料集是同一份，569 位病人、30 項腫瘤特徵、良性/惡性標籤 |
| Digits（手寫數字，8×8 小圖） | 13 | `sklearn.datasets.load_digits`，套件內建 |
| pydicom 內建 CT 範例 | 8, 22 | `pydicom.data.get_testdata_file()`，套件內建的真實 DICOM 檔案，免下載 |
| 合成資料（模擬 radiomics 高維度小樣本） | 15 | `sklearn.datasets.make_regression` 產生，用來模擬「300 個特徵、80 位病人」的情境 |

**想換成 Kaggle 上的真實醫學影像資料集？** 多數程式碼把資料載入獨立成一個函式（例如 `load_data()`、`load_imbalanced_mnist()`），只要照著同樣的輸入輸出格式，把資料來源換成你從 Kaggle 下載好的資料夾即可。第 7 課（分割任務類型）的程式碼註解中特別列出了對應的 Kaggle 資料集建議（2018 Data Science Bowl 細胞核實例分割）。下載與登入 Kaggle 帳號需要你自己完成，我們不會替你處理帳密或下載檔案。

## 執行方式

先確認 conda 環境已啟用（見上方「環境設定」），每個資料夾都是獨立的，進去之後直接執行對應的 `.py` 檔即可，例如：

```bash
conda activate hibio-tutorial
cd lessons/16-mlp
python mlp_demo.py
```

程式會把訓練結果（準確率、圖表等）印在終端機，部分課會把圖片存到 `tutorial/outputs/`。

## 目錄結構

```
tutorial/
├── index.html              # 教學網頁（22 堂課的說明、圖示、練習）
├── requirements.txt         # pip 套件清單（在 conda 環境裡用 pip install -r 這個檔案）
├── lessons/
│   ├── 01-holdout/
│   ├── 02-kfold-cv/
│   ├── ...
│   ├── 20-vit-gradcam/
│   ├── 21-transfer-learning/
│   └── 22-data-augmentation-medical/
├── data/                    # 資料集下載快取（.gitignore，不會進版控）
└── outputs/                 # 程式產生的圖表（.gitignore，不會進版控）
```
