import cv2
import numpy as np
import easyocr

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # 左上
    rect[2] = pts[np.argmax(s)]   # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
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

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    M_inv = cv2.getPerspectiveTransform(dst, rect)   # 反向矩陣，避免 np.linalg.inv 出錯
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, M, M_inv, maxWidth, maxHeight

# 初始化 OCR
reader = easyocr.Reader(['en'], gpu=False)

# --- Step 1: 標準答案處理 ---
standard_img = cv2.imread(r".\static\standard.png")
standard_gray = cv2.cvtColor(standard_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(standard_gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_approx = None
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        area = cv2.contourArea(approx)
        if area > max_area:
            max_area = area
            best_approx = approx

if best_approx is not None:
    standard_warped, _, _, _, _ = four_point_transform(standard_img, best_approx.reshape(4,2))
    standard_results = reader.readtext(standard_warped)

    # 存標準答案的「位置」（用中心點）
    standard_positions = []
    for (bbox, text, conf) in standard_results:
        pts = np.array(bbox, dtype=np.int32)
        cx = int(np.mean(pts[:,0]))
        cy = int(np.mean(pts[:,1]))
        standard_positions.append((cx, cy, pts))
else:
    print("❌ Step 1: Standard Answer Processing ---")
    exit()

# --- Step 2: 影片處理 ---
cap = cv2.VideoCapture(r".\static\wrong.mp4")
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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
            if area > max_area:
                max_area = area
                best_approx = approx

    if best_approx is not None:
        warped, M, M_inv, w, h = four_point_transform(frame, best_approx.reshape(4,2))
        ocr_results = reader.readtext(warped)

        detected_positions = []
        for (bbox, text, confidence) in ocr_results:
            pts = np.array(bbox, dtype=np.int32)
            cx = int(np.mean(pts[:,0]))
            cy = int(np.mean(pts[:,1]))
            detected_positions.append((cx, cy, pts, confidence))

        # --- Step 3: Compare positions ---
        for std_cx, std_cy, std_pts in standard_positions:
            found_match = False
            for det_cx, det_cy, det_pts, det_conf in detected_positions:
                dist = np.sqrt((std_cx - det_cx)**2 + (std_cy - det_cy)**2)
                if dist < 50:  # Tolerance (pixels)
                    found_match = True
                    # Warped window
                    cv2.polylines(warped, [det_pts], isClosed=True, color=(0,255,0), thickness=2)
                    cv2.putText(warped, f"correct ({det_conf:.2f})",
                                (det_pts[0][0], det_pts[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    # Frame window (map back to original image)
                    det_pts_h = cv2.perspectiveTransform(det_pts.reshape(-1,1,2).astype(np.float32), M_inv)
                    det_pts_h = det_pts_h.reshape(-1,2).astype(int)
                    cv2.polylines(frame, [det_pts_h], isClosed=True, color=(0,255,0), thickness=2)
                    cv2.putText(frame, f"correct ({det_conf:.2f})",
                                (det_pts_h[0][0], det_pts_h[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    break

            # 🔹 Only draw red box if standard point is inside the main rectangle
            if not found_match:
                if cv2.pointPolygonTest(best_approx, (std_cx, std_cy), False) >= 0:
                    # Warped window
                    cv2.polylines(warped, [std_pts], isClosed=True, color=(0,0,255), thickness=2)
                    cv2.putText(warped, "error", (std_pts[0][0], std_pts[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                    # Frame window (map back to original image)
                    std_pts_h = cv2.perspectiveTransform(std_pts.reshape(-1,1,2).astype(np.float32), M_inv)
                    std_pts_h = std_pts_h.reshape(-1,2).astype(int)
                    cv2.polylines(frame, [std_pts_h], isClosed=True, color=(0,0,255), thickness=2)
                    cv2.putText(frame, "error", (std_pts_h[0][0], std_pts_h[0][1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow("Warped + OCR Compare", warped)
        cv2.drawContours(frame, [best_approx], -1, (0,255,0), 3)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
