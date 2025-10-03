# OCR 缺陷檢測系統

基於 EasyOCR 的視覺檢測系統，用於即時識別和驗證產品標籤文字位置。

## 功能特點

- 即時視訊流 OCR 文字識別
- 透視變換校正
- 位置偏移檢測（IoU 比對）
- 自動資料庫記錄
- 檢測區域可調整

## 環境需求

### Python 版本
- Python 3.10 或以上

### 核心套件版本
```
opencv-python==4.8.1.78
numpy==1.24.3
easyocr==1.7.0
pyodbc==5.0.1
```

### 系統需求
- **GPU 支援**（建議）：CUDA 11.x + cuDNN
- **CPU 模式**：將 `ocr_reader = easyocr.Reader(['en'], gpu=True)` 改為 `gpu=False`

## 安裝步驟

### 1. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安裝相依套件
```bash
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install easyocr==1.7.0
pip install pyodbc==5.0.1
```

### 3. 資料庫驅動安裝
安裝 Microsoft ODBC Driver 17 for SQL Server：
- **Windows**: [下載連結](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)
- **Linux**: 參考 [官方文件](https://docs.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server)

### 4. PaddleOCR（備選方案）
本專案使用 EasyOCR，若需要 PaddleOCR：
```bash
pip install paddleocr==2.7.0
pip install paddlepaddle-gpu  # GPU版本
# 或
pip install paddlepaddle  # CPU版本
```

## 專案結構

```
DEFECTDETECTION_NUMBER/
├── analysis/
│   └── temp/              # 儲存檢測圖片
├── reference.jpg          # 標準答案圖片
├── sample.mp4            # 測試影片
└── main_camera_Homography_EasyOCR.py
```

## 使用方法

### 1. 準備標準答案圖片
將標準答案圖片命名為 `reference.jpg` 放在專案根目錄

### 2. 設定資料庫連線
修改程式碼中的 `DATABASE_CONFIG`：
```python
DATABASE_CONFIG = {
    'server': 'your_server_ip',
    'database': 'OCR',
    'username': 'your_username',
    'password': 'your_password',
    'driver': '{ODBC Driver 17 for SQL Server}'
}
```

### 3. 執行程式
```bash
python main_camera_Homography_EasyOCR.py
```

### 4. 操作快捷鍵
- `q`: 退出程式
- `a`: 偵測線左移
- `d`: 偵測線右移
- `s`: 手動觸發 debug 儲存
- `r`: 重置 debug 計數器

## 輸出說明

### 資料庫欄位
- **Value**: OCR 識別文字
- **Confidence**: 識別信心度 (0-1)
- **Status**: 位置狀態 (1=clear, 5=misaligned)
- **Image_base64**: ROI 區域 base64 編碼
- **Spacing**: IoU 數值
- **Filename**: 完整畫面檔名

### 儲存位置
- 完整畫面：`./analysis/temp/{serial_number}.jpg`
- ROI base64：存入資料庫

## 參數調整

```python
conf_threshold = 0.5      # OCR 信心度閾值
OCR_INTERVAL = 1          # OCR 執行間隔 (frames)
LINE_POSITION_RATIO = 0.4 # 偵測線位置 (0-1)
STANDARD_WIDTH = 320      # 標準化寬度
STANDARD_HEIGHT = 240     # 標準化高度
```

## 常見問題

### 1. CUDA 記憶體不足
將 `gpu=True` 改為 `gpu=False`

### 2. 資料庫連線失敗
檢查防火牆設定和 SQL Server 允許遠端連線

### 3. 找不到 ODBC Driver
重新安裝對應版本的 ODBC Driver

