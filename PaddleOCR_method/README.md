# OCR 缺陷檢測系統 (PaddleOCR 版本)

基於 PaddleOCR 的視覺檢測系統，用於即時識別和驗證產品標籤文字位置。

## 功能特點

- 即時視訊流 OCR 文字識別
- 透視變換校正與標準化
- 位置偏移檢測（IoU 比對）
- 自動位置驗證（單次檢測）
- Base64 圖像輸出

## 環境需求

### Python 版本
- Python 3.8 或以上

### 核心套件版本
```
opencv-python==4.8.1.78
numpy==1.24.3
paddlepaddle-gpu==2.6.0  # GPU版本
# 或
paddlepaddle==2.6.0      # CPU版本
paddleocr==2.7.0
```

### 系統需求
- **GPU 支援**（建議）：CUDA 11.x + cuDNN 8.x
- **CPU 模式**：使用 paddlepaddle CPU 版本

## 安裝步驟

### 1. 建立虛擬環境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 2. 安裝 PaddlePaddle

**GPU 版本**（推薦，需要 NVIDIA GPU）：
```bash
pip install paddlepaddle-gpu==2.6.0 -i https://mirror.baidu.com/pypi/simple
```

**CPU 版本**：
```bash
pip install paddlepaddle==2.6.0 -i https://mirror.baidu.com/pypi/simple
```

### 3. 安裝其他相依套件
```bash
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install paddleocr==2.7.0
```

### 4. 下載 PaddleOCR 模型
將預訓練模型放置在專案目錄：
```
PP-OCRv5_mobile_det/    # 文字檢測模型
PP-OCRv5_mobile_rec/    # 文字識別模型
```

模型下載來源：[PaddleOCR 官方模型庫](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md)

## 專案結構

```
DEFECTDETECTION_NUMBER/
├── PP-OCRv5_mobile_det/   # 檢測模型
├── PP-OCRv5_mobile_rec/   # 識別模型
├── reference.jpg          # 標準答案圖片
├── sample.mp4            # 測試影片
└── main_vedio_Fixedpoint_PaddleOCR.py
```

## 使用方法

### 1. 準備標準答案圖片
將標準答案圖片命名為 `reference.jpg` 放在專案根目錄

### 2. 執行程式
```bash
python main_vedio_Fixedpoint_PaddleOCR.py
```

### 3. 操作快捷鍵
- `q`: 退出程式

## 核心參數設定

```python
conf_threshold = 0.5         # OCR 信心度閾值
STANDARD_WIDTH = 400         # 標準化寬度
STANDARD_HEIGHT = 300        # 標準化高度
defect_margin = 30          # 檢測區域邊界緩衝
```

### IoU 位置檢測
```python
threshold=0.6  # IoU 閾值，可在 calculate_iou 中調整
```

## 輸出說明

### JSON 輸出格式
```json
{
    1: {
        "value": "ABC123",
        "confidence": 0.997,
        "status": "clear",        // "clear" 或 "misaligned"
        "image_base64": "..."
    }
}
```

### 欄位說明
- **value**: OCR 識別的文字（最高出現次數）
- **confidence**: 識別信心度 (0-1)
- **status**: 位置對齊狀態
  - `clear`: 位置正確（IoU ≥ 0.6）
  - `misaligned`: 位置偏移（IoU < 0.6）
- **image_base64**: ROI 區域 base64 編碼

## 位置檢測機制

### 標準化流程
1. 載入標準答案圖片 → 透視變換 → 保存 warped 圖像
2. 檢測當前物件 → 透視變換 → 獲得 warped 圖像
3. **兩張圖片都統一 resize 到 400x300**
4. 在標準化尺寸下重新 OCR
5. 比對相同文字的 bbox 座標
6. 計算 IoU 判斷位置是否正確

### 單次檢測策略
- 每個檢測會話只在**首次檢測到文字時**進行一次位置檢查
- 檢查完成後記錄狀態，直到離開檢測區域才輸出
- 避免重複計算，提升效能

## PaddleOCR 配置

```python
PaddleOCR(
    use_textline_orientation=True,  # 支援文字方向檢測
    lang="en",                      # 英文識別
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir=r".\PP-OCRv5_mobile_det",
    text_recognition_model_dir=r".\PP-OCRv5_mobile_rec",
)
```

詳細配置參考：[PaddleOCR 官方文檔](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html)

## 常見問題

### 1. CUDA 記憶體不足
安裝 CPU 版本的 PaddlePaddle：
```bash
pip uninstall paddlepaddle-gpu
pip install paddlepaddle==2.6.0
```

### 2. 模型載入失敗
確認模型路徑正確，模型檔案完整下載

### 3. 位置檢測總是 clear
- 檢查 IoU 閾值設定（預設 0.6）
- 確認標準答案圖片正確載入
- 查看 console 輸出的座標比對資訊

### 4. OCR 識別率低
- 調整 `conf_threshold` 降低閾值
- 檢查影片/圖片清晰度
- 確認光照條件適當

## 效能優化建議

1. **使用 GPU**：識別速度提升 5-10 倍
2. **調整標準化尺寸**：更大的尺寸 (如 640x480) 識別更準確但速度較慢
3. **調整檢測間隔**：可加入 frame skip 機制

## 版本差異

| 特性 | PaddleOCR 版本 | EasyOCR 版本 |
|------|---------------|--------------|
| 中文支援 | ✅ 優秀 | ⚠️ 一般 |
| 英文識別 | ✅ 優秀 | ✅ 優秀 |
| 速度 | ⚡ 快 | 🐢 較慢 |
| 安裝複雜度 | ⚠️ 較高 | ✅ 簡單 |
| GPU 需求 | 建議 | 建議 |

## 授權
請參考 PaddleOCR 的 Apache 2.0 授權

## 相關資源
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddlePaddle 官網](https://www.paddlepaddle.org.cn/)
- [模型下載](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/models_list.md)