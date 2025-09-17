import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import time
from collections import Counter
import base64

# 初始化 PaddleOCR
# https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html#1-ocr
ocr_reader = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    text_detection_model_dir=r".\PP-OCRv5_mobile_det",
    text_recognition_model_dir=r".\PP-OCRv5_mobile_rec",
)

conf_threshold = 0.5
frame_count = 0
start_time = time.time()
last_ocr_time = 0
text_counter = Counter()
text_conf_dict = {}  # 記錄每個文字最高 confidence和座標
serial_number = 1
out_of_area = True  # 一開始假設不在檢測區
defect_margin = 30  # 邊界緩衝
last_roi_image = None  # 保存最後一次的ROI圖像

# 標準化尺寸
STANDARD_WIDTH = 400
STANDARD_HEIGHT = 300


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    if maxWidth <= 0 or maxHeight <= 0:
        return None
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def normalize_image_size(image, target_width=STANDARD_WIDTH, target_height=STANDARD_HEIGHT):
    """將圖片標準化到指定尺寸"""
    if image is None:
        return None
    return cv2.resize(image, (target_width, target_height))

def bbox_to_minmax(bbox_points):
    """將PaddleOCR的bbox點轉換為[x_min, y_min, x_max, y_max]格式"""
    x_coords = [point[0] for point in bbox_points]
    y_coords = [point[1] for point in bbox_points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def normalize_bbox_coordinates(bbox, original_width, original_height, target_width=STANDARD_WIDTH, target_height=STANDARD_HEIGHT):
    """將bbox座標從原始尺寸正規化到目標尺寸"""
    x_min, y_min, x_max, y_max = bbox
    
    # 計算縮放比例
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    # 正規化座標
    normalized_bbox = [
        int(x_min * scale_x),
        int(y_min * scale_y),
        int(x_max * scale_x),
        int(y_max * scale_y)
    ]
    
    return normalized_bbox

def run_ocr_and_collect_text(warped_img, conf_threshold=0.5):                
    results = {}
    try:
        ocr_results = ocr_reader.predict(warped_img)
        #print(f"🔍 ocr_results: {ocr_results}")  # ← 看看有沒有偵測到任何東西
        for page in ocr_results:
            rec_texts = page.get('rec_texts', [])
            rec_scores = page.get('rec_scores', [])
            rec_polys = page.get('rec_polys', [])  # 四點座標
            #print(f"📦 texts={rec_texts}, scores={rec_scores}, boxes={rec_polys}")
            for poly, text, score in zip(rec_polys, rec_texts, rec_scores):
                if score >= conf_threshold:
                    clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                    if clean_text:
                        text_counter[clean_text] += 1
                        text_conf_dict[clean_text] = max(text_conf_dict.get(clean_text, 0.0), float(score))
                        results[clean_text] = {
                            "score": float(score),
                            "bbox": bbox_to_minmax(poly)
                        }
    except Exception as e:
        print("OCR error on warped:", e)
    return results

def image_to_base64(image):
    """将OpenCV图像转换为base64字符串"""
    try:
        # 将图像编码为JPEG格式
        _, buffer = cv2.imencode('.jpg', image)
        # 转换为base64字符串
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def calculate_iou(reference_warped_img, current_warped_img, reference_text, current_text, threshold=0.9):
    """
    計算兩個矩形的IoU (Intersection over Union)重疊率
    先將兩張圖片都標準化到400x300，然後重新OCR獲取標準化後的座標進行比較
    
    參數:
    - reference_warped_img: 標準答案的warped圖片
    - current_warped_img: 當前檢測的warped圖片  
    - reference_text: 標準答案的文字
    - current_text: 當前檢測的文字
    - threshold: IoU閾值
    """
    try:
        # 1. 將兩張圖片都標準化到400x300
        ref_normalized = cv2.resize(reference_warped_img, (STANDARD_WIDTH, STANDARD_HEIGHT))
        curr_normalized = cv2.resize(current_warped_img, (STANDARD_WIDTH, STANDARD_HEIGHT))
        
        # 2. 對標準化後的圖片重新進行OCR
        ref_ocr_results = ocr_reader.predict(ref_normalized)
        curr_ocr_results = ocr_reader.predict(curr_normalized)
        
        # 3. 從標準答案中找到對應文字的bbox
        ref_bbox = None
        for page in ref_ocr_results:
            rec_texts = page.get('rec_texts', [])
            rec_polys = page.get('rec_polys', [])
            for poly, text in zip(rec_polys, rec_texts):
                clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                if clean_text == reference_text:
                    ref_bbox = bbox_to_minmax(poly)
                    break
            if ref_bbox:
                break
                
        # 4. 從當前檢測中找到對應文字的bbox
        curr_bbox = None
        for page in curr_ocr_results:
            rec_texts = page.get('rec_texts', [])
            rec_polys = page.get('rec_polys', [])
            for poly, text in zip(rec_polys, rec_texts):
                clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                if clean_text == current_text:
                    curr_bbox = bbox_to_minmax(poly)
                    break
            if curr_bbox:
                break
        
        if ref_bbox is None or curr_bbox is None:
            print(f"❌ 無法找到文字座標: ref_bbox={ref_bbox}, curr_bbox={curr_bbox}")
            return 0.0, False
        
        # 5. 計算IoU
        box1, box2 = ref_bbox, curr_bbox
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])  
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            print(f"📦 無重疊: ref_bbox={ref_bbox}, curr_bbox={curr_bbox}")
            return 0.0, False
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0.0
        is_aligned = iou >= threshold
        
        print(f"📏 標準化後座標比較({STANDARD_WIDTH}x{STANDARD_HEIGHT}):")
        print(f"   標準答案 '{reference_text}': {ref_bbox}")
        print(f"   當前檢測 '{current_text}': {curr_bbox}")
        print(f"   IoU={iou:.3f}, 對齊={'✅' if is_aligned else '❌'}")
        
        return iou, is_aligned
        
    except Exception as e:
        print(f"❌ calculate_iou error: {e}")
        return 0.0, False

# 全局變數存儲標準答案圖片和文字
reference_warped_img = None
reference_text = None

# 當前檢測會話的位置記錄
current_session_position_checked = False
current_session_position_status = "clear"
current_session_warped_img = None

def load_reference_image(ref_path):
    """載入標準答案圖片並記錄warped圖片和文字"""
    global reference_warped_img, reference_text
    
    try:
        ref_image = cv2.imread(ref_path)
        if ref_image is None:
            print(f"❌ Cannot load reference image: {ref_path}")
            return False

        # 找標準答案的矩形並做透視變換
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_approx = None
        valid_rectangles = 0
        
        for i, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(approx)
 
            if len(approx) == 4:
                valid_rectangles += 1
                if area > 5000:
                    if area > max_area:
                        max_area = area
                        best_approx = approx.reshape(4, 2).astype(np.float32)
  
        if best_approx is None:
            print("❌ Cannot find rectangle in reference image")
            return False

        # 透視變換
        ref_warped = four_point_transform(ref_image, best_approx)
        if ref_warped is None:
            print("❌ Perspective transform failed")
            return False

        print(f"📐 標準圖透視變換後尺寸: {ref_warped.shape[:2]}")
        
        # 保存warped圖片
        reference_warped_img = ref_warped.copy()
        
        # OCR標準答案獲取文字
        ocr_texts = run_ocr_and_collect_text(ref_warped, conf_threshold)
        
        if len(ocr_texts) != 0:
            # 保存第一個檢測到的文字
            reference_text = list(ocr_texts.keys())[0]
            print(f"📍 標準答案文字: '{reference_text}'")
            return True
        return False

    except Exception as e:
        print(f"❌ Error loading reference image: {e}")
        import traceback
        traceback.print_exc()
        return False

# 載入標準答案
reference_image_path = r".\reference.jpg"  # 請修改為你的標準答案圖片路徑
reference_loaded = load_reference_image(reference_image_path)
print(f"✅ Reference image loaded: {reference_loaded}")
if reference_loaded:
    print(f"📝 Reference text: '{reference_text}'")

# 影片讀取
video_path = r".\sample.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    height, width = frame.shape[:2]

    right_half = frame[:, width//2:].copy()
    gray = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_approx = None
    best_rect = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h) if h != 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    if area > max_area:
                        max_area = area
                        best_approx = approx.reshape(4, 2).astype(np.float32)
                        best_rect = (x, y, w, h)

    if best_rect is not None:
        x, y, w, h = best_rect
        rect_center_x = x + w / 2
        rect_center_y = y + h / 2

        defect_x_min = defect_margin
        defect_x_max = right_half.shape[1] - defect_margin
        defect_y_min = defect_margin
        defect_y_max = right_half.shape[0] - defect_margin

        inside_area = (
            defect_x_min <= rect_center_x <= defect_x_max and
            defect_y_min <= rect_center_y <= defect_y_max
        )

        if inside_area:
            if out_of_area:
                text_counter = Counter()
                text_conf_dict.clear()
                # 重置當前會話的位置檢查狀態
                current_session_position_checked = False
                current_session_position_status = "clear"
                current_session_warped_img = None
                print("📄 Re-entered defect area → reset counter and position status")
                out_of_area = False
        else:
            # 仍在畫面但超出 defect area 不重置，等消失再處理
            pass

        warped = None
        if best_approx is not None:
            warped = four_point_transform(right_half, best_approx)
        else:
            warped = right_half[y:y+h, x:x+w].copy()

        current_ocr_texts = run_ocr_and_collect_text(warped, conf_threshold)
        
        # 檢查是否需要進行位置檢測（只在第一次檢測到文字且有標準答案時進行）
        if (not current_session_position_checked and 
            current_ocr_texts and 
            reference_warped_img is not None and 
            reference_text is not None):
            
            # 取得當前檢測到的第一個文字進行位置檢查
            first_detected_text = list(current_ocr_texts.keys())[0]
            print(f"🎯 首次檢測到文字 '{first_detected_text}'，進行位置檢查...")
            
            iou_value, is_aligned = calculate_iou(
                reference_warped_img, 
                warped, 
                reference_text, 
                first_detected_text, 
                threshold=0.6
            )
            
            current_session_position_status = "clear" if is_aligned else "misaligned"
            current_session_position_checked = True
            current_session_warped_img = warped.copy()  # 保存這一幀的圖片
            
            print(f"✅ 位置檢查完成: IoU={iou_value:.3f}, Status={current_session_position_status}")
            print("💡 本次會話不再重複進行位置檢查")

        if warped is not None:
            roi_display = warped.copy()
            if current_ocr_texts:
                
                text_line = " | ".join(current_ocr_texts.keys())  # 用 key 串起來顯示
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text_line, font, font_scale, font_thickness)
                padding = 10
                cv2.rectangle(roi_display, (0, 0), (roi_display.shape[1], text_height + padding * 2), (255, 255, 255), -1)
                cv2.rectangle(roi_display, (0, 0), (roi_display.shape[1], text_height + padding * 2), (0, 0, 0), 2)
                cv2.putText(roi_display, text_line, (padding, text_height + padding),
                            font, font_scale, (0, 0, 0), font_thickness)
            
            # 保存包含OCR文本叠加的標準化ROI圖像
            last_roi_image = roi_display.copy()
            cv2.imshow("ROI + OCR Compare", roi_display)

        if best_approx is not None:
            pts_int = best_approx.astype(int)
            pts_int[:, 0] += width // 2
            cv2.polylines(frame, [pts_int], True, (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x + width//2, y), (x + width//2 + w, y + h), (255, 0, 0), 2)

        top3 = text_counter.most_common(3)
        y0 = 30
        for i, (word, count) in enumerate(top3):
            y = y0 + i * 30
            cv2.putText(frame, f"{word}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # 沒有找到矩形 → 視為離開
        if not out_of_area:
            # 在離開前輸出 JSON
            if text_counter:
                top1, count = text_counter.most_common(1)[0]
                confidence = text_conf_dict.get(top1, 0.0)
                
                # 使用已經檢查過的位置狀態
                position_status = current_session_position_status
                print(f"📋 使用已記錄的位置狀態: {position_status}")
                
                # 將最後的ROI圖像轉換為base64（優先使用位置檢查時保存的圖片）
                image_base64 = ""
                roi_image_to_save = current_session_warped_img if current_session_warped_img is not None else last_roi_image
                if roi_image_to_save is not None:
                    image_base64 = image_to_base64(roi_image_to_save)

                result_json = {
                    serial_number: {
                        "value": top1,
                        "confidence": round(confidence, 3),
                        "status": position_status,
                        #"bbox": current_bbox,  # 添加座標資訊
                        "image_base64": image_base64
                    }
                }
                print("📤 JSON result:", result_json)
                serial_number += 1

            print("⬅️ Lost detection → mark as out of area")
            out_of_area = True

    # 畫檢測區框線
    cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 255), 2)

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