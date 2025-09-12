import cv2
import numpy as np
import easyocr

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

# 初始化 OCR
reader = easyocr.Reader(['en'], gpu=False)

# 追蹤字串的變數
current_tracked_texts = []  # 當前追蹤的字串
text_history = {}  # 字串歷史記錄 {text: [positions, confidences, frame_count]}

print("開始處理影片，追蹤字串移動...")

# --- 影片處理 ---
cap = cv2.VideoCapture(r".\static\wrong.mp4")
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    
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
        ocr_results = reader.readtext(rotated_warped)

        detected_positions = []
        detected_boxes_in_rotated_view = []  # 儲存在旋轉視圖中的實際OCR框
        
        current_frame_texts = []
        
        for (bbox, text, confidence) in ocr_results:
            # OCR結果的座標是在旋轉圖像中的，直接使用
            pts_rotated = np.array(bbox, dtype=np.int32)
            
            # 調整座標回到原始warped圖像的座標系統用於比較
            adjusted_bbox = adjust_ocr_coordinates_for_rotation(bbox, h)
            pts = np.array(adjusted_bbox, dtype=np.int32)
            cx = int(np.mean(pts[:,0]))
            cy = int(np.mean(pts[:,1]))
            
            text_clean = text.strip()
            current_frame_texts.append(text_clean)
            
            detected_positions.append((cx, cy, pts, confidence, text_clean))
            detected_boxes_in_rotated_view.append((pts_rotated, text_clean, confidence))
            
            # 更新字串歷史
            if text_clean not in text_history:
                text_history[text_clean] = {'positions': [], 'confidences': [], 'last_seen': frame_count}
            
            text_history[text_clean]['positions'].append((cx, cy))
            text_history[text_clean]['confidences'].append(confidence)
            text_history[text_clean]['last_seen'] = frame_count

        # 更新當前追蹤的字串
        current_tracked_texts = current_frame_texts.copy()

        # 在旋轉視圖中繪製檢測到的文字（綠框）
        for pts_rotated, text, confidence in detected_boxes_in_rotated_view:
            cv2.polylines(rotated_warped, [pts_rotated], isClosed=True, color=(0,255,0), thickness=2)
            cv2.putText(rotated_warped, f"{text} ({confidence:.2f})",
                        (pts_rotated[0][0], pts_rotated[0][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 檢查是否有字串消失（切換到下一個）
        disappeared_texts = []
        for text in text_history.keys():
            if text not in current_frame_texts and frame_count - text_history[text]['last_seen'] > 5:
                disappeared_texts.append(text)

        # 分析消失的字串
        for text in disappeared_texts:
            history = text_history[text]
            avg_confidence = np.mean(history['confidences'])
            total_frames = len(history['positions'])
            
            print(f"字串 '{text}' 已消失 - 共出現 {total_frames} 幀，平均信心度: {avg_confidence:.2f}")
            
            # 進行三重檢查分析
            try:
                num = int(text)
                print(f"  數字: {num}")
                
                # 這裡可以加入你的三重檢查邏輯
                # 1. 檢查是否漏印
                # 2. 檢查是否重複 
                # 3. 檢查是否跳號
                
            except:
                print(f"  非數字文字: {text}")

        # 在Frame上標示追蹤區域
        cv2.drawContours(frame, [best_approx], -1, (0,255,0), 3)
        
        # 顯示當前追蹤的字串信息
        info_text = f"Frame: {frame_count}, Tracking: {len(current_tracked_texts)} texts"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        if current_tracked_texts:
            texts_display = ", ".join(current_tracked_texts)
            cv2.putText(frame, f"Current: {texts_display}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # 顯示旋轉後的圖像作為主要比較視窗
        cv2.imshow("Warped + OCR Compare", rotated_warped)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n=== 追蹤結果總結 ===")
print(f"總共追蹤到 {len(text_history)} 個不同的字串:")
for text, history in text_history.items():
    avg_conf = np.mean(history['confidences'])
    print(f"'{text}': 出現 {len(history['positions'])} 次，平均信心度: {avg_conf:.2f}")