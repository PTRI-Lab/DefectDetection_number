#0901
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import time
from collections import Counter

# ----------------------------
# 初始化 PaddleOCR
# ----------------------------
ocr_reader = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir=r".\PP-OCRv5_mobile_det",
    text_recognition_model_dir=r".\PP-OCRv5_mobile_rec",
)

conf_threshold = 0.5
pattern = re.compile(r'[A-Za-z0-9]+')

# ----------------------------
# 影片讀取
# ----------------------------
video_path = r".\static\wrong.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

frame_count = 0
start_time = time.time()

# OCR 控制
last_ocr_time = 0
ocr_interval = 1.0  # 每秒辨識一次

# 累積結果
text_counter = Counter()

# 上一張紙的位置
prev_rect = None
new_page_threshold = 50  # 判斷新紙的差異門檻

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # 取得影片尺寸，只處理右半邊
    height, width = frame.shape[:2]
    right_half = frame[:, width//2:]

    # 找最大矩形 (只在右半邊)
    gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 5000:
                rect = cv2.boundingRect(approx)
                aspect_ratio = rect[2] / rect[3]
                if 0.3 < aspect_ratio < 3.0:
                    if area > max_area:
                        max_area = area
                        best_rect = rect

    if best_rect is not None:
        x, y, w, h = best_rect
        # 從右半邊提取 ROI
        roi = right_half[y:y+h, x:x+w]

        # --- 判斷是否新的一張紙 ---
        if prev_rect is not None:
            diff = sum(abs(a - b) for a, b in zip(best_rect, prev_rect))
            if diff > new_page_threshold:
                text_counter = Counter()  # 歸零
                print("📄 New page detected, reset counter")
        prev_rect = best_rect

        # --- ROI 旋轉90度 ---
        roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

        # --- 每秒 OCR 一次 ---
        current_ocr_texts = []  # 存儲當前幀的OCR結果
        if time.time() - last_ocr_time > ocr_interval:
            last_ocr_time = time.time()
            results = ocr_reader.predict(roi_rotated)

            for page in results:
                rec_texts = page.get('rec_texts', [])
                rec_scores = page.get('rec_scores', [])
                rec_boxes = page.get('rec_boxes', [])
                for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                    if score >= conf_threshold:
                        clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                        if clean_text:
                            text_counter[clean_text] += 1
                            current_ocr_texts.append(clean_text)

        # --- ROI 視窗顯示 OCR 結果 ---
        roi_display = roi_rotated.copy()
        
        # 在ROI視窗最上方顯示當前辨識到的文字（黑體白色背景）
        if current_ocr_texts:
            # 將所有辨識到的文字合併為一行
            text_line = " | ".join(current_ocr_texts)
            
            # 計算文字尺寸
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text_line, font, font_scale, font_thickness)
            
            # 在最上方畫白色背景矩形
            padding = 10
            cv2.rectangle(roi_display, (0, 0), (roi_display.shape[1], text_height + padding * 2), (255, 255, 255), -1)
            
            # 畫黑色邊框
            cv2.rectangle(roi_display, (0, 0), (roi_display.shape[1], text_height + padding * 2), (0, 0, 0), 2)
            
            # 顯示黑色文字
            cv2.putText(roi_display, text_line, (padding, text_height + padding),
                       font, font_scale, (0, 0, 0), font_thickness)

        cv2.imshow("ROI + OCR Compare", roi_display)

        # 在原始frame上標示檢測區域（調整座標到完整frame）
        rect_x = x + width//2  # 加上左半邊的寬度
        rect_y = y
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + w, rect_y + h), (255, 0, 0), 2)

        # --- Top3 顯示在 Frame ---
        top3 = text_counter.most_common(3)
        y0 = 30
        for i, (word, count) in enumerate(top3):
            y = y0 + i * 30
            cv2.putText(frame, f"{word}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 在frame上畫出右半邊分界線
    cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 255), 2)
    cv2.putText(frame, "Detection Area", (width//2 + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 顯示 FPS 與 Frame 數
    cv2.putText(frame, f"Frame: {frame_count}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()