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

# 上一張紙的位置和最高頻文字
prev_rect = None
prev_top_text = None  # 儲存上一次的第一名文字

def is_rect_on_left(current_rect, previous_rect):
    """判斷當前矩形是否在上一個矩形的左邊"""
    if previous_rect is None:
        return False
    
    # 比較左上角X座標
    current_left_x = current_rect[0]
    previous_left_x = previous_rect[0]
    if current_left_x < previous_left_x: # 新點在舊點的左邊
        if abs(previous_left_x-current_left_x)<=100: # 新舊點差距不大
            return True
    return False

def has_same_top_text(current_texts, prev_top_text):
    """檢查當前OCR結果是否包含上一次的第一名文字"""
    if not prev_top_text or not current_texts:
        return False
    
    # 直接檢查上一次第一名文字是否在當前OCR結果中
    return prev_top_text in current_texts

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # 找最大矩形
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
        roi = frame[y:y+h, x:x+w]

        # --- ROI 旋轉90度 ---
        roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

        # --- 每幀都執行 OCR ---
        ocr_results = []
        current_texts = set()
        
        print(f"🔍 Running OCR on frame {frame_count}...")
        results = ocr_reader.predict(roi_rotated)

        for page in results:
            rec_texts = page.get('rec_texts', [])
            rec_scores = page.get('rec_scores', [])
            rec_boxes = page.get('rec_boxes', [])
            for text, score, box in zip(rec_texts, rec_scores, rec_boxes):
                if score >= conf_threshold:
                    clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                    if clean_text:
                        current_texts.add(clean_text)
                        ocr_results.append((box, clean_text, score))

        # --- 判斷是否為同一張紙 ---
        should_reset = True  # 預設重置
        
        if prev_rect is not None and prev_top_text is not None and len(current_texts) !=0 :
            # 判斷矩形是否在左邊
            rect_on_left = is_rect_on_left(best_rect, prev_rect)
            
            # 檢查當前OCR結果是否包含上一次的第一名文字
            has_top_text = has_same_top_text(current_texts, prev_top_text)
            
            print(f"🔍 Frame {frame_count} Debug:")
            print(f"   rect_on_left={rect_on_left}")
            print(f"   has_top_text={has_top_text}")
            print(f"   current_texts={current_texts}")
            print(f"   prev_top_text='{prev_top_text}'")
            print(f"   prev_rect={prev_rect}")
            print(f"   best_rect={best_rect}")
            
            # 如果新矩形在左邊 AND OCR辨識到上一次第一名文字 -> 不重置
            if rect_on_left : #has_top_text:
                should_reset = False
                print("✅ Same page: rectangle moved left and still has previous top text")
            else:
                should_reset = True
                print("❌ New page: different position or previous top text not found:({0},{1})".format(current_texts, prev_top_text))
        else:
            # 第一次檢測，不重置
            should_reset = False
            print("📄 First page detected")
        
        # 重置或繼續累積
        if should_reset:
            text_counter = Counter()
            print("🔄 Counter reset for new page")
        else:
            print("✅ Continue counting on same page")
        
        # 累積文字計數
        for _, text, _ in ocr_results:
            text_counter[text] += 1
        
        # 更新記錄
        prev_rect = best_rect
        # 更新上一次的第一名文字
        if text_counter:
            prev_top_text = text_counter.most_common(1)[0][0]

        # --- Top3 顯示在 Frame ---
        top3 = text_counter.most_common(3)
        y0 = 30
        for i, (word, count) in enumerate(top3):
            y = y0 + i * 30
            cv2.putText(frame, f"{word}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- ROI 視窗顯示 OCR 結果 ---
        roi_display = roi_rotated.copy()
        
        # 在視窗最上方顯示OCR結果文字

        # 白色背景
        cv2.rectangle(roi_display, (10, 0), (400, 35), (255, 255, 255), -1)
        # 黑色文字
        cv2.putText(roi_display, f"{top3[0][0]} ({score:.2f})", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)


        cv2.imshow("ROI + OCR Compare", roi_display)

        # 顯示當前辨識到的文字數量
        cv2.putText(frame, f"Current texts: {len(current_texts)}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

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