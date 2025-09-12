import cv2
import numpy as np
import easyocr
import time


def order_points(pts):
    """
    將四個點按照順序排列：左上、右上、右下、左下
    透過計算點的座標和與差值來判斷各點位置
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def four_point_transform(image, pts):
    """
    執行四點透視變換，將傾斜的四邊形矯正為矩形
    計算目標矩形的寬度和高度，並建立變換矩陣
    返回：矯正後圖像、正向變換矩陣、反向變換矩陣、寬度、高度
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    M_inv = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M, M_inv, maxWidth, maxHeight

def rotate_image_90_clockwise(image):
    """順時針旋轉90度"""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_coordinates_to_rotated_view(pts, original_width):
    """
    將原始warped座標轉換到旋轉後的座標系統
    順時針90度旋轉: (x, y) -> (original_width - y, x)
    """
    rotated_pts = []
    for point in pts:
        x, y = point
        new_x = original_width - y
        new_y = x
        rotated_pts.append([new_x, new_y])
    return rotated_pts

def adjust_ocr_coordinates_for_rotation(bbox, original_height):
    """
    調整OCR座標以適應順時針90度旋轉
    旋轉前: (x, y) -> 旋轉後: (y, original_height - x)
    """
    adjusted_bbox = []
    for point in bbox:
        x, y = point
        new_x = y
        new_y = original_height - x
        adjusted_bbox.append([new_x, new_y])
    return adjusted_bbox

def preprocess_for_ocr(image):
    """
    OCR 前處理：
    1. 灰階化
    2. 自適應二值化
    3. 形態學操作去噪
    4. 對比度增強 (CLAHE)
    5. 放大圖片
    6. 膨脹
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 對比度增強
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # 二值化
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 25, 15
    )

    # 形態學處理去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 先侵蝕，將白色小圓點移除
    erode = cv2.erode(morph, kernel, iterations=2)     
    # 膨脹
    dilated = cv2.dilate(erode, kernel, iterations=2)

    return dilated

# 初始化 OCR
reader = easyocr.Reader(['en'], gpu=False)

# --- Step 1: 影片處理 ---
cap = cv2.VideoCapture(r".\static\wrong.mp4")
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

#第幾偵
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_approx = None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 5000:  # 最小面積閾值，過濾小輪廓
                # 檢查是否接近矩形（長寬比合理）
                rect = cv2.boundingRect(approx)
                aspect_ratio = rect[2] / rect[3]  # 寬/高
                if 0.3 < aspect_ratio < 3.0:  # 長寬比在合理範圍內
                    if area > max_area:
                        max_area = area
                        best_approx = approx

    if best_approx is not None:
        warped, M, M_inv, w, h = four_point_transform(frame, best_approx.reshape(4,2))
        
        # 旋轉影片幀90度後進行OCR
        rotated_warped = rotate_image_90_clockwise(warped)
        # 前處理
        preprocessed = preprocess_for_ocr(rotated_warped)
        # OCR
        ocr_results = reader.readtext(preprocessed)

        detected_positions = []
        detected_boxes_in_rotated_view = []  # 儲存在旋轉視圖中的實際OCR框
        
        for (bbox, text, confidence) in ocr_results:
            # OCR結果的座標是在旋轉圖像中的，直接使用
            pts_rotated = np.array(bbox, dtype=np.int32)
            
            # 調整座標回到原始warped圖像的座標系統用於比較
            adjusted_bbox = adjust_ocr_coordinates_for_rotation(bbox, h)
            pts = np.array(adjusted_bbox, dtype=np.int32)
            cx = int(np.mean(pts[:,0]))
            cy = int(np.mean(pts[:,1]))
            
            detected_positions.append((cx, cy, pts, confidence, text))
            detected_boxes_in_rotated_view.append((pts_rotated, text, confidence))
            #print(f"Detected text: '{text}' at center ({cx}, {cy}) confidence: {confidence:.2f}")

        # --- Step 2: Compare positions ---

        # 在旋轉視圖中繪製檢測到的文字（綠框）
        for pts_rotated, text, confidence in detected_boxes_in_rotated_view:
            cv2.polylines(preprocessed, [pts_rotated], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(preprocessed, f"{text} ({confidence:.2f})",
                        (pts_rotated[0][0], pts_rotated[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 顯示旋轉後的圖像作為主要比較視窗
        cv2.imshow("Warped + OCR Compare", preprocessed)
        cv2.drawContours(frame, [best_approx], -1, (0,255,0), 3)

    # 在 frame 左上角顯示 Frame 計數 & FPS
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()