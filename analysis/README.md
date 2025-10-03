```markdown
# 影像對齊分析系統

自動監控影像檔案、進行透視校正、偵測 ROI 區域，並計算與基準影像的對齊偏移量。

## 功能特色

- **自動監控資料夾**：持續監看輸入目錄，自動處理新增的影像檔案
- **透視校正**：自動偵測四邊形輪廓並進行透視變換，將影像校正為正視圖
- **智慧 ROI 偵測**：
  - 綠色外框偵測（支援 HSV 嚴格/寬鬆模式及灰階備援）
  - 內部深色面板偵測
  - 面板內文字區域精確定位
- **對齊分析**：計算影像相對於基準的垂直/水平偏移及旋轉狀態
- **資料庫整合**：自動更新 SQL Server 資料庫中的偏移數據
- **偵錯視覺化**：輸出標註影像及中間處理遮罩，便於問題排查

## 系統需求

### Python 環境
- Python 3.7+

### 必要套件
```bash
pip install opencv-python numpy pyodbc
```

### 資料庫
- SQL Server（需安裝 ODBC Driver 17 或 18）
- 資料庫名稱：`OCR`
- 資料表：`UV`（需包含 `Filename`, `v_shift`, `h_shift`, `rotate` 欄位）

## 專案結構

```
project/
├── main.py                 # 主程式
├── temp/                   # 輸入目錄（監控此資料夾）
├── temp_out/               # 輸出目錄
│   ├── debug/              # 偵錯影像（遮罩、中間結果）
│   ├── alignment_results.csv  # CSV 結果記錄
│   ├── _baseline.json      # 基準數據
│   └── analysis_*.jpg      # 標註後的分析影像
└── reference3.jpg          # 基準參考影像（需放置於 ../reference3.jpg）
```

## 設定

### 1. 資料庫設定

編輯 `DATABASE_CONFIG` 字典：

```python
DATABASE_CONFIG = {
    'server': 'your_server_ip',      # SQL Server IP/hostname
    'database': 'OCR',
    'username': 'your_username',
    'password': 'your_password',
    'driver': '{ODBC Driver 17 for SQL Server}'  # Windows
    # Linux 使用: '{ODBC Driver 18 for SQL Server}'
}
```

### 2. 路徑設定

```python
INPUT_DIR = "temp"          # 輸入監控目錄
OUTPUT_DIR = "temp_out"     # 輸出目錄
REFERENCE_PATH = r"../reference3.jpg"  # 基準影像路徑
```

### 3. 偵測參數調整

```python
# HSV 綠色範圍（針對外框偵測）
GREEN_STRICT = ((40, 60, 40), (85, 255, 255))  # 嚴格模式
GREEN_RELAX  = ((30, 40, 30), (95, 255, 255))  # 寬鬆模式

# 透視變換邊距
margin = 30  # 校正後影像的白邊寬度
```

## 使用方式

### 啟動程式

```bash
python main.py
```

### 輸入檔案要求

- **檔名格式**：純數字 + `.jpg` 擴展名（例如：`001.jpg`, `123.jpg`）
- **影像內容**：包含綠色外框及內部面板的待測影像
- **放置位置**：將影像放入 `temp/` 目錄

### 處理流程

1. 程式啟動後會清空 `temp_out/` 並初始化 CSV
2. 首次運行會從 `reference3.jpg` 建立基準數據（`_baseline.json`）
3. 持續監控 `temp/` 目錄：
   - 偵測新影像 → 透視校正 → 輸出至 `temp_out/`
   - 分析 ROI → 計算偏移量 → 更新資料庫 & CSV
4. 每個影像會生成：
   - `{filename}.jpg`：校正後影像
   - `analysis_{filename}.jpg`：標註影像（顯示 ROI 及偏移值）
   - `debug/{filename}_*.png`：偵錯遮罩

### 停止程式

按 `Ctrl+C` 中斷執行

## 輸出說明

### CSV 檔案（`alignment_results.csv`）

| 欄位 | 說明 |
|------|------|
| filename | 檔案名稱 |
| v_shift | 垂直偏移量（像素，正值表示向下偏移） |
| h_shift | 水平偏移量（像素，負值表示向左偏移） |
| rotate | 旋轉標記（0=未旋轉, 1=已旋轉） |

### 標註影像

分析影像會顯示：
- **綠色矩形**：偵測到的綠色外框（Panel ROI）
- **藍色矩形**：偵測到的文字區域（Text ROI）
- **數值標註**：top/left/right/bottom 間距（像素）

### 偵錯遮罩（`debug/` 目錄）

- `*_frame_strict.png`：嚴格模式綠色遮罩
- `*_frame_relax.png`：寬鬆模式綠色遮罩
- `*_frame_fallback.png`：灰階備援遮罩
- `*_panel_mask.png`：面板偵測遮罩
- `*_text_bin.png`：二值化文字遮罩
- `*_text_dil.png`：膨脹後的文字遮罩

## 演算法說明

### 1. 透視校正

- 使用 Canny 邊緣偵測 + 輪廓近似找出四邊形
- 備援方案：`minAreaRect` 找出最小包圍矩形
- 四點透視變換並強制輸出為橫向影像

### 2. ROI 偵測

**綠色外框**：
1. HSV 色彩空間綠色範圍篩選
2. 形態學閉運算去除雜訊
3. 找出最大輪廓的外接矩形

**內部面板**：
1. 在外框內使用 Otsu 閾值 + 低亮度閾值偵測深色區域
2. 形態學處理 + 侵蝕避免邊界干擾
3. 找出最大深色區域並內縮邊界

**文字區域**：
1. CLAHE 對比度增強
2. 自適應閾值 + Otsu 閾值聯集
3. 水平方向膨脹連接字符
4. 多候選區域評分（中心位置、寬度、高度、長寬比）
5. 投影收緊邊界

### 3. 對齊分析

- 計算基準影像的 `top_ratio` 和 `left_ratio`
- 待測影像與基準比較：
  - **v_shift**：垂直間距差異
  - **h_shift**：水平間距差異（負值表示向左）
  - **rotate**：文字高度差異超過 ±6px 標記為已旋轉

## 故障排除

### 無法偵測到 ROI

1. 檢查 `debug/` 目錄的遮罩影像
2. 調整 HSV 範圍參數（`GREEN_STRICT`, `GREEN_RELAX`）
3. 確認影像包含明確的綠色外框及深色面板

### 資料庫連線失敗

1. 確認 SQL Server 啟用 TCP/IP 連線
2. 檢查防火牆設定（預設 1433 port）
3. 驗證 ODBC Driver 版本：
   ```bash
   # Windows
   odbcad32
   
   # Linux
   odbcinst -j
   ```

### 透視校正失敗

- 確保影像中有清晰的四邊形輪廓
- 調整 Canny 參數（`cv2.Canny` 的閾值）
- 檢查影像解析度及品質

## 注意事項

- 程式啟動會清空 `temp_out/` 目錄及 CSV 檔案
- 檔案名稱必須為純數字（如 `001.jpg`），其他格式會被忽略
- 基準影像 `reference3.jpg` 必須先準備好並正確定位
- 資料庫連線失敗不會中斷程式，但不會更新資料庫

## 日誌說明

```
[2025-01-15 14:32:01] [CLEAN] temp_out/ cleared and CSV reinitialized.
[2025-01-15 14:32:05] [RECTIFIED] 001.jpg → temp_out/001.jpg
[2025-01-15 14:32:06] [BASELINE] 已從 ../reference3.jpg 建立 baseline.
[2025-01-15 14:32:07] [DB] Updated record for Filename='001.jpg' ...
[2025-01-15 14:32:07] [ANALYZED] 001.jpg → v_shift=5, h_shift=-3, rotate=0
```

