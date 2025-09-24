# OCR 影像識別系統 README

## 專案簡介

這是一個基於 OpenCV 的實時影像文字識別系統，提供兩種 OCR 引擎版本：
- **EasyOCR 版本**：支援多平台，適合跨平台部署
- **PaddleOCR 版本**：提供更高精度，但僅支援 Windows 平台

### 主要功能
- 即時影片中矩形區域的文字識別
- 與參考圖像進行位置比對
- 將識別結果存儲到 SQL Server 資料庫（EasyOCR 版本）
- 支援調試模式，提供詳細的比對結果

## 系統需求

### 共同需求
- Python 3.7+
- Windows/Linux/macOS (依版本而定)

### EasyOCR 版本
- 支援所有平台：Windows、Linux、macOS
- 可選支援 GPU 的環境（用於加速 OCR 處理）

### PaddleOCR 版本
- **僅支援 Windows 平台**
- 不支援 Linux 環境
- 建議使用 GPU 以獲得最佳性能

## 安裝步驟

### 1. 安裝 Python 套件

請根據您要使用的 OCR 引擎選擇對應的安裝方式：

#### EasyOCR 版本（支援所有平台）

```bash
# 更新 pip 到最新版本
pip install --upgrade pip

# 安裝核心套件
pip install opencv-python
pip install easyocr
pip install numpy
pip install pyodbc

# 如果需要 GPU 支援，請安裝 CUDA 相關套件
pip install torch torchvision torchaudio
```

#### PaddleOCR 版本（僅 Windows）

⚠️ **注意：PaddleOCR 版本不支援 Linux 系統**

```bash
# 更新 pip 到最新版本
pip install --upgrade pip

# 安裝核心套件
pip install opencv-python
pip install paddlepaddle  # 或 paddlepaddle-gpu（如有 GPU）
pip install paddleocr
pip install numpy

# 注意：PaddleOCR 版本不包含資料庫功能，無需安裝 pyodbc
```

#### 下載模型檔案（PaddleOCR 版本）

PaddleOCR 版本需要預先下載模型檔案：

1. 創建模型目錄：
```bash
mkdir PP-OCRv5_mobile_det
mkdir PP-OCRv5_mobile_rec
```

2. 從 [PaddleOCR Model Zoo](https://paddlepaddle.github.io/PaddleOCR/latest/models/PP-OCRv4/overview.html) 下載對應模型文件並放入相應目錄

### 2. 資料庫驅動程式安裝

由於程式使用 SQL Server，需要安裝 ODBC 驅動程式：

**Windows:**
1. 下載並安裝 [Microsoft ODBC Driver 18 for SQL Server](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

**Linux:**
```bash
# Ubuntu/Debian
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18

# CentOS/RHEL
curl https://packages.microsoft.com/config/rhel/8/prod.repo > /etc/yum.repos.d/mssql-release.repo
yum remove unixODBC-utf16 unixODBC-utf16-devel
ACCEPT_EULA=Y yum install -y msodbcsql18
```

**macOS:**
```bash
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
HOMEBREW_NO_ENV_FILTERING=1 ACCEPT_EULA=Y brew install msodbcsql18
```

## 執行方式

### 1. 準備必要檔案

根據您選擇的版本準備對應檔案：

#### EasyOCR 版本
- `reference3.jpg` - 參考圖像檔案
- `sample3.mp4` - 待處理的影片檔案
- 程式檔案：`ocr_detection_easyocr.py`

#### PaddleOCR 版本
- `reference.jpg` - 參考圖像檔案
- `sample.mp4` - 待處理的影片檔案
- 程式檔案：`ocr_detection_paddleocr.py`
- 模型目錄：`PP-OCRv5_mobile_det/` 和 `PP-OCRv5_mobile_rec/`

### 2. 設定資料庫連線（僅 EasyOCR 版本）

修改程式中的資料庫連線設定：

```python
DATABASE_CONFIG = {
    'server': '您的SQL Server IP',
    'database': 'OCR',
    'username': '您的使用者名稱',
    'password': '您的密碼',
    'driver': '{ODBC Driver 18 for SQL Server}'
}
```

### 3. 執行程式

#### EasyOCR 版本
```bash
python ocr_detection_easyocr.py
```

#### PaddleOCR 版本
```bash
python ocr_detection_paddleocr.py
```

### 4. 程式操作說明

程式執行後會顯示即時影像視窗，支援以下按鍵操作：

- **`q`** - 退出程式
- **`a`** - 向左移動檢測線
- **`d`** - 向右移動檢測線
- **`s`** - 手動保存除錯資訊
- **`r`** - 重置除錯計數器

### 5. 程式輸出

#### EasyOCR 版本
程式會將結果存儲到 SQL Server 資料庫，並在控制台顯示：

```
[RESULT] JSON result:Value=識別文字,Confidence=0.95,Status=1,Image_base64=iVBOR...,Spacing=0.75
```

#### PaddleOCR 版本
程式會在控制台輸出 JSON 格式結果：

```json
{
    1: {
        "value": "識別到的文字",
        "confidence": 0.95,
        "status": "clear",
        "image_base64": "base64編碼的圖像"
    }
}
```

## OCR 引擎比較

### EasyOCR vs PaddleOCR

| 特性 | EasyOCR | PaddleOCR |
|------|---------|-----------|
| **平台支援** | Windows, Linux, macOS | **僅 Windows** |
| **語言支援** | 80+ 種語言 | 80+ 種語言 |
| **精確度** | 高 | **更高** |
| **安裝複雜度** | 簡單 | 中等（需下載模型） |
| **記憶體使用** | 適中 | 較高 |
| **資料庫整合** | 完整支援 | 無 |
| **API 易用性** | 非常簡單 | 簡單 |

## EasyOCR 介紹

### 什麼是 EasyOCR？

EasyOCR 是一個基於深度學習的開源 OCR（光學字符識別）程式庫，具有以下特點：

1. **多語言支援** - 支援超過 80 種語言
2. **易於使用** - 簡潔的 API 介面
3. **高精度** - 基於 CRAFT 文字檢測和 CRNN 文字識別
4. **GPU 加速** - 支援 CUDA 加速處理

### 在本專案中的應用

```python
# 初始化 EasyOCR 讀取器
ocr_reader = easyocr.Reader(['en'], gpu=True)

# 進行文字識別
results = ocr_reader.readtext(image)
# 返回格式：(bbox, text, confidence)
```

### EasyOCR 的優勢

1. **準確性高** - 對於印刷體文字有很高的識別準確率
2. **處理能力強** - 能處理各種角度和光照條件下的文字
3. **邊界框檢測** - 同時提供文字位置資訊
4. **信心度評分** - 為每個識別結果提供置信度分數

### 支援的語言

EasyOCR 支援多種語言，常用的包括：
- 英文 (`en`)
- 中文簡體 (`ch_sim`)
- 中文繁體 (`ch_tra`)
- 日文 (`ja`)
- 韓文 (`ko`)

在本專案中，我們使用英文識別：
```python
# EasyOCR
ocr_reader = easyocr.Reader(['en'], gpu=True)

# PaddleOCR  
ocr_reader = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec"
)
```

## PaddleOCR 介紹

### 什麼是 PaddleOCR？

PaddleOCR 是百度開源的超輕量級 OCR 系統，具有以下特點：

1. **超輕量級** - 模型體積小，推理速度快
2. **高精確度** - 基於 PP-OCRv5 模型，識別精度更高
3. **多語言支援** - 支援 80+ 種語言的識別
4. **產業級應用** - 已在多個實際場景中驗證

### 在本專案中的應用

```python
# 初始化 PaddleOCR
ocr_reader = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir=r".\PP-OCRv5_mobile_det",
    text_recognition_model_dir=r".\PP-OCRv5_mobile_rec"
)

# 進行文字識別
results = ocr_reader.predict(image)
```

### PaddleOCR 的優勢

1. **速度快** - 針對移動端優化，處理速度更快
2. **精度高** - PP-OCRv5 模型在多個數據集上表現優異
3. **資源占用低** - 記憶體和計算資源需求相對較小
4. **靈活配置** - 支援自定義模型路徑和參數

### 重要限制

⚠️ **PaddleOCR 版本僅支援 Windows 平台**

- 在 Linux 環境下可能出現兼容性問題
- 建議 Linux 用戶使用 EasyOCR 版本

## 故障排除

### 常見問題

1. **ODBC 驅動程式錯誤（EasyOCR 版本）**
   ```
   解決方案：請確保已安裝正確版本的 ODBC 驅動程式
   ```

2. **EasyOCR 初始化失敗**
   ```bash
   # 可能是 GPU 記憶體不足，可以關閉 GPU 支援
   ocr_reader = easyocr.Reader(['en'], gpu=False)
   ```

3. **PaddleOCR 在 Linux 上無法運行**
   ```
   解決方案：PaddleOCR 版本不支援 Linux，請使用 EasyOCR 版本
   ```

4. **PaddleOCR 模型載入失敗**
   ```
   解決方案：確認模型目錄路徑正確，且已下載完整的模型檔案
   ```

5. **影片檔案無法開啟**
   ```
   解決方案：檢查影片檔案路徑和格式是否正確
   ```

6. **資料庫連線失敗（EasyOCR 版本）**
   ```
   解決方案：檢查網路連線、資料庫設定和防火牆規則
   ```

### 平台選擇建議

- **Windows 用戶**：可選擇任一版本，PaddleOCR 提供更高精度
- **Linux 用戶**：必須使用 EasyOCR 版本
- **macOS 用戶**：建議使用 EasyOCR 版本
- **需要資料庫整合**：使用 EasyOCR 版本

### 效能優化建議

1. **GPU 支援** - 如果有 NVIDIA GPU，啟用 GPU 支援可以顯著提升處理速度
2. **影像預處理** - 適當的影像預處理可以提高 OCR 準確率
3. **檢測間隔** - 調整 `OCR_INTERVAL` 參數以平衡處理速度和準確性

## 技術支援

如果遇到問題，請檢查：
1. Python 版本是否符合需求
2. 所有必要的套件是否正確安裝
3. 資料庫連線設定是否正確
4. 檔案路徑是否存在

## 版本資訊

### EasyOCR 版本
- Python: 3.7+
- OpenCV: 4.x
- EasyOCR: 1.6+
- PyODBC: 4.x
- 支援平台: Windows, Linux, macOS

### PaddleOCR 版本
- Python: 3.7+
- OpenCV: 4.x
- PaddleOCR: 2.6+
- PaddlePaddle: 2.4+
- 支援平台: **僅 Windows**

## 檔案結構

```
project/
├── ocr_detection_easyocr.py    # EasyOCR 版本主程式
├── ocr_detection_paddleocr.py  # PaddleOCR 版本主程式
├── reference.jpg               # PaddleOCR 版本參考圖片
├── reference3.jpg              # EasyOCR 版本參考圖片
├── sample.mp4                  # PaddleOCR 版本測試影片
├── sample3.mp4                 # EasyOCR 版本測試影片
├── PP-OCRv5_mobile_det/       # PaddleOCR 文字檢測模型
├── PP-OCRv5_mobile_rec/       # PaddleOCR 文字識別模型
└── README.md                   # 本文件
```

## 授權

本專案僅供學習和研究使用。
