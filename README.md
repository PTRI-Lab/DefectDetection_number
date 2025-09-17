
## 🧩 環境需求與安裝

* **Python 版本**：`Python 3.10.11`
* **必須套件**（可用 `pip` 安裝）：

```bash
pip install opencv-python numpy paddleocr
```

> 📌 安裝 `paddleocr` 時會自動安裝 `paddlepaddle`，如遇 GPU 版本需求，可參考 [PaddleOCR 官方安裝指南](https://www.paddleocr.ai/latest/installation/)

---

### ▶️ 執行方式

1. 將 `main_vedio_Fixedpoint_PaddleOCR.py` 放在專案根目錄
2. 準備以下兩個檔案，並放在與程式同一層目錄：

   * `sample.mp4`：要分析的影片檔
   * `reference.jpg`：作為比對用的參考影像
3. 在終端機執行：

```bash
python main_vedio_Fixedpoint_PaddleOCR.py
```

4. 執行後會：

   * 讀取影片右半區域的矩形區塊
   * 對 ROI 進行透視變換與 OCR
   * 與參考影像中 OCR 文字與位置（IoU）比對
   * 在畫面上顯示 OCR 結果與比對狀態，並於物件離開檢測區後輸出 JSON 結果於終端機

---

### 📁 所需資料檔案

| 檔案名稱            | 說明               |
| --------------- | ---------------- |
| `sample.mp4`    | 待檢測的影片           |
| `reference.jpg` | 參考基準影像（含矩形與標準文字） |

---

### 📦 PaddleOCR 模型下載

此程式使用 **PP-OCRv5** 的輕量模型，可於 PaddleOCR 官網下載：

* 官方模型清單與使用說明：
  🔗 [PaddleOCR 官方說明](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html#1-ocr)

* 需下載以下兩個模型，並放置於程式同目錄：

  ```
  ./PP-OCRv5_mobile_det
  ./PP-OCRv5_mobile_rec
  ```

程式會透過以下參數載入：

```python
text_detection_model_name="PP-OCRv5_mobile_det",
text_recognition_model_name="PP-OCRv5_mobile_rec",
text_detection_model_dir="./PP-OCRv5_mobile_det",
text_recognition_model_dir="./PP-OCRv5_mobile_rec",
```

---
