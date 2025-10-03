import cv2
import numpy as np
import easyocr
import re
import time
import pyodbc
import base64

from collections import Counter
#from datetime import datetime


# Database configuration
DATABASE_CONFIG = {
    'server': '...',  # Replace with your actual IP
    'database': 'OCR',
    'username': 'ptridev',
    'password': '...',  # Replace with actual password
    'driver': '{ODBC Driver 17 for SQL Server}'  # or '{SQL Server}' for older versions 。linux: '{ODBC Driver 18 for SQL Server}'
}

def init_database():
    """Initialize database connection and clear existing data for testing"""
    try:
        conn_str = (
            f"DRIVER={DATABASE_CONFIG['driver']};"
            f"SERVER={DATABASE_CONFIG['server']};"
            f"DATABASE={DATABASE_CONFIG['database']};"
            f"UID={DATABASE_CONFIG['username']};"
            f"PWD={DATABASE_CONFIG['password']};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
            "Connection Timeout=30;"
        )

        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # Clear existing data for testing (using the UV table your colleague created)
        clear_data_sql = "DELETE FROM UV"
        cursor.execute(clear_data_sql)
        
        conn.commit()
        print(f"[DB] Database connected and UV table cleared for testing")
        print(f"[DB] Connected to: {DATABASE_CONFIG['server']}/OCR")
        
        return conn
        
    except Exception as e:
        print(f"[DB ERROR] Failed to connect to database: {e}")
        return None

def insert_to_database(conn, data):
    """Insert OCR result to database"""
    if conn is None:
        print("[DB ERROR] No database connection available")
        return False
        
    try:
        cursor = conn.cursor()
        
        # Note: This assumes UV table has columns matching your JSON structure
        # You may need to adjust column names based on actual table schema
        insert_sql = """
        INSERT INTO UV ( Value, Confidence, Status, Image_base64,Spacing,Filename )
        VALUES (  ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(insert_sql, (
            data['Value'],
            data['Confidence'],
            data['Status'],
            data['Image_base64'],
            data['Spacing'],
            data['Filename']
        ))
        
        conn.commit()
        print(f"[DB] Data inserted to UV table:  Value='{data['Value']}', Status={data['Status']}")
        return True
        
    except Exception as e:
        print(f"[DB ERROR] Failed to insert data to UV table: {e}")
        print(f"[DB ERROR] You may need to check UV table schema matches the data structure")
        return False


# Initialize database connection at startup
db_connection = init_database()

# Initialize EasyOCR
# Supports English, can add other languages if needed, e.g., ['en', 'ch_sim'] for Chinese and English
ocr_reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if no GPU

conf_threshold = 0.5
frame_count = 0
start_time = time.time()
last_ocr_time = 0
text_counter = Counter()
text_conf_dict = {}  # Record highest confidence and coordinates for each text
serial_number = 1
out_of_area = True  # Initially assume not in detection area
defect_margin = 30  # Boundary buffer
last_roi_image = None  # Save last ROI image
iou_value = 0.0  # define IOU
OCR_INTERVAL = 1 # Perform OCR every N frames
# Standard size
STANDARD_WIDTH = 320 #400
STANDARD_HEIGHT = 240 #300

# Middle line position setting (adjustable)
# 0.5 = center of screen, 0.6 = right bias, 0.4 = left bias
LINE_POSITION_RATIO = 0.4  # Adjust this value to change line position

# DEBUG: Add debug flag
DEBUG_MODE = True
debug_image_counter = 0

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
    """Normalize image to specified size"""
    if image is None:
        return None
    return cv2.resize(image, (target_width, target_height))

def easyocr_bbox_to_minmax(bbox_points):
    """Convert EasyOCR bbox points to [x_min, y_min, x_max, y_max] format"""
    # EasyOCR returns bbox format as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    x_coords = [point[0] for point in bbox_points]
    y_coords = [point[1] for point in bbox_points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def normalize_bbox_coordinates(bbox, original_width, original_height, target_width=STANDARD_WIDTH, target_height=STANDARD_HEIGHT):
    """Normalize bbox coordinates from original size to target size"""
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate scaling ratios
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    
    # Normalize coordinates
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
        # EasyOCR readtext method returns (bbox, text, confidence)
        ocr_results = ocr_reader.readtext(warped_img)
        for (bbox, text, confidence) in ocr_results:
            if confidence >= conf_threshold:
                #print(f"[OCR] Original text: '{text}' (confidence: {confidence:.3f})")
                # Modified: Replace non-alphanumeric characters with space, then remove spaces
                clean_text = re.sub(r'[^A-Za-z0-9]', '', text.replace(' ', ''))
                #print(f"[OCR] Cleaned text: '{clean_text}'")
                if clean_text:
                    text_counter[clean_text] += 1
                    text_conf_dict[clean_text] = max(text_conf_dict.get(clean_text, 0.0), float(confidence))
                    results[clean_text] = {
                        "score": float(confidence),
                        "bbox": easyocr_bbox_to_minmax(bbox)
                    }
    except Exception as e:
        print("OCR error on warped:", e)
    return results

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        # Encode image as JPEG format
        _, buffer = cv2.imencode('.jpg', image)
        #_, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])  # 70% 品質
        # Convert to base64 string
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def draw_bbox_on_image(image, bbox, color=(0, 255, 0), thickness=2, label=""):
    """Draw bbox on image"""
    img_copy = image.copy()
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, thickness)
    
    if label:
        # Add text label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Background rectangle
        cv2.rectangle(img_copy, (x_min, y_min - text_height - 5), 
                     (x_min + text_width, y_min), color, -1)
        cv2.putText(img_copy, label, (x_min, y_min - 5), 
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return img_copy

def save_debug_comparison(ref_img, curr_img, ref_bbox, curr_bbox, ref_text, curr_text, iou_value):
    """Save debug comparison image"""
    global debug_image_counter
    
    try:
        # Normalize both images to same size
        ref_normalized = cv2.resize(ref_img, (STANDARD_WIDTH, STANDARD_HEIGHT))
        curr_normalized = cv2.resize(curr_img, (STANDARD_WIDTH, STANDARD_HEIGHT))
        
        # Draw bbox on images
        ref_with_bbox = draw_bbox_on_image(ref_normalized, ref_bbox, 
                                         color=(0, 255, 0), 
                                         label=f"REF: {ref_text}")
        curr_with_bbox = draw_bbox_on_image(curr_normalized, curr_bbox, 
                                          color=(255, 0, 0), 
                                          label=f"CURR: {curr_text}")
        
        # Create side-by-side comparison
        comparison = np.hstack([ref_with_bbox, curr_with_bbox])
        
        # Add title information
        title_height = 60
        title_img = np.zeros((title_height, comparison.shape[1], 3), dtype=np.uint8)
        title_img[:] = (50, 50, 50)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        title_text = f"IOU: '{iou_value}, 'CURR: '{curr_text}'"
        text_size = cv2.getTextSize(title_text, font, font_scale, font_thickness)[0]
        text_x = (comparison.shape[1] - text_size[0]) // 2
        text_y = (title_height + text_size[1]) // 2
        
        cv2.putText(title_img, title_text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        # Combine final image
        final_img = np.vstack([title_img, comparison])
        
        # Save image
        debug_image_counter += 1
        filename = f"debug_comparison_{debug_image_counter:03d}.jpg"
        #cv2.imwrite(filename, final_img)
        #print(f"[DEBUG] Debug image saved: {filename}")
        
        # Also display in window
        cv2.imshow("Debug Comparison", final_img)
        
        return filename
        
    except Exception as e:
        print(f"[ERROR] Error saving debug image: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_iou(current_warped_img, current_text, threshold=0.9):
    """
    Calculate IoU between current image and reference answer
    - current_warped_img: Current warped ROI
    - current_text: Currently detected text (only for logging, doesn't affect position judgment)
    - threshold: IoU threshold
    
    Note: Position checking is independent of text content, mainly checks the position of the first/largest text box detected by OCR
    """
    global reference_warped_img, reference_text, reference_bbox
    try:
        if reference_warped_img is None or reference_bbox is None:
            print("[ERROR] Reference not properly loaded, cannot calculate IoU")
            return 0.0, False
        '''
        # DEBUG: Print original image sizes
        print(f"[DEBUG] Reference image size: {reference_warped_img.shape}")
        print(f"[DEBUG] Current image size: {current_warped_img.shape}")
        print(f"[DEBUG] Reference bbox: {reference_bbox}")
        '''
        # Only do OCR on current warped image
        curr_normalized = cv2.resize(current_warped_img, (STANDARD_WIDTH, STANDARD_HEIGHT))
        curr_ocr_results = ocr_reader.readtext(curr_normalized)

        #print(f"[OCR] Current image OCR result count: {len(curr_ocr_results)}")
        
        if len(curr_ocr_results) == 0:
            print("[ERROR] No text detected in current image")
            return 0.0, False
        
        # Independent position judgment: Use bbox with largest area or highest confidence
        best_bbox = None
        best_area = 0
        best_confidence = 0
        #chosen_method = ""
        chosen_text = ""
        
        for (bbox, text, confidence) in curr_ocr_results:
            clean_text = re.sub(r'[^A-Za-z0-9]', '', text.replace(' ', ''))
            curr_bbox_coords = easyocr_bbox_to_minmax(bbox)
            bbox_area = (curr_bbox_coords[2] - curr_bbox_coords[0]) * (curr_bbox_coords[3] - curr_bbox_coords[1])
            
            #print(f"- Detected text: '{clean_text}' (original: '{text}'), confidence: {confidence:.3f}, area: {bbox_area}")
            
            # Strategy 1: Prioritize bbox with largest area (usually main text)
            if bbox_area > best_area:
                best_bbox = curr_bbox_coords
                best_area = bbox_area
                best_confidence = confidence
                chosen_method = "largest_area"
                chosen_text = clean_text
            # Strategy 2: If areas are similar, choose higher confidence
            elif abs(bbox_area - best_area) < (best_area * 0.1) and confidence > best_confidence:
                best_bbox = curr_bbox_coords
                best_area = bbox_area
                best_confidence = confidence
                chosen_method = "highest_confidence"
                chosen_text = clean_text
        
        #print(f"[BBOX] Selected bbox: {best_bbox} (strategy: {chosen_method}, text: '{chosen_text}', area: {best_area})")

        if best_bbox is None:
            print("[ERROR] Cannot determine best bbox")
            return 0.0, False

        # Calculate IoU
        box1, box2 = reference_bbox, best_bbox
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        '''
        print(f"[IoU] IoU calculation details:")
        print(f"   Reference bbox: {box1}")
        print(f"   Current bbox: {box2}")
        print(f"   Intersection coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        '''
        if x2 < x1 or y2 < y1:
            print(f"[IoU] No overlap: ref={box1}, curr={box2}")
            return 0.0, False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        '''
        print(f"   Intersection area: {intersection}")
        print(f"   Reference area: {area1}")
        print(f"   Current area: {area2}")
        print(f"   Union area: {union}")
        '''
        iou = intersection / union if union > 0 else 0.0
        is_aligned = iou >= threshold

        #print(f"[IoU] IoU={iou:.3f}, Aligned={'YES' if is_aligned else 'NO'}")
        #print(f"[RESULT] Statistics text='{current_text}', Position check text='{chosen_text}' (independent judgment)")
        # DEBUG: Always save debug comparison image
        if DEBUG_MODE:
            save_debug_comparison(
                reference_warped_img, current_warped_img,
                reference_bbox, best_bbox,
                reference_text, f"{chosen_text}", round(iou, 3)  # IoU will be calculated below
            )

        return iou, is_aligned

    except Exception as e:
        print(f"[ERROR] calculate_iou error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, False

# Global variables to store reference image and text
reference_warped_img = None
reference_text = None
reference_bbox = None  # Record reference text bbox

# Current detection session position record
current_session_position_checked = False
current_session_position_status = 1
current_session_warped_img = None

# Track whether rectangle crosses the middle line
rect_crossed_line = False
last_rect_x = None  # Record rectangle x coordinate from previous frame

def load_reference_image(ref_path):
    """Load reference image and record warped image, text and bbox"""
    global reference_warped_img, reference_text, reference_bbox

    try:
        ref_image = cv2.imread(ref_path)
        if ref_image is None:
            print(f"[ERROR] Cannot load reference image: {ref_path}")
            return False

        # Find rectangle contours
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(ref_gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        best_approx = None
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            area = cv2.contourArea(approx)

            if len(approx) == 4 and area > 5000:
                if area > max_area:
                    max_area = area
                    best_approx = approx.reshape(4, 2).astype(np.float32)

        if best_approx is None:
            print("[ERROR] Cannot find rectangle in reference image")
            return False

        # Perspective transform
        ref_warped = four_point_transform(ref_image, best_approx)
        if ref_warped is None:
            print("[ERROR] Perspective transform failed")
            return False

        #print(f"[DEBUG] Reference image warped size: {ref_warped.shape[:2]}")

        # DEBUG: Save reference warped version
        #cv2.imwrite("debug_reference_warped.jpg", ref_warped)
        #print("[DEBUG] Reference warped image saved: debug_reference_warped.jpg")

        # Save warped image (standardized size)
        reference_warped_img = cv2.resize(ref_warped, (STANDARD_WIDTH, STANDARD_HEIGHT))

        # OCR once, get first text + bbox
        ocr_texts = run_ocr_and_collect_text(reference_warped_img, conf_threshold)
        if len(ocr_texts) > 0:
            reference_text = list(ocr_texts.keys())[0]
            reference_bbox = list(ocr_texts.values())[0]["bbox"]  # Store reference bbox
            #print(f"[REF] Reference text: '{reference_text}', bbox={reference_bbox}")
            
            # DEBUG: Save reference image with bbox annotation
            #ref_with_bbox = draw_bbox_on_image(reference_warped_img, reference_bbox,color=(0, 255, 0), label=f"REF: {reference_text}")
            #cv2.imwrite("debug_reference_with_bbox.jpg", ref_with_bbox)
            #print("[DEBUG] Reference image with bbox saved: debug_reference_with_bbox.jpg")
            
            return True
        else:
            print("[ERROR] Reference image OCR found no text")
            return False

    except Exception as e:
        print(f"[ERROR] Error loading reference image: {e}")
        import traceback
        traceback.print_exc()
        return False
        
# Load reference answer
reference_image_path = r"./reference.jpg"  # Please modify to your reference image path
reference_loaded = load_reference_image(reference_image_path)
print(f"[INIT] Reference image loaded: {reference_loaded}")
if reference_loaded:
    print(f"[INIT] Reference text: '{reference_text}'")

# Video reading
cap = cv2.VideoCapture("sample.mp4")
if not cap.isOpened():
    print("[ERROR] Cannot open video")
    exit()

#print("[DEBUG] DEBUG mode enabled, debug images will be saved to current directory")
#print("   Press 's' key to force save current frame debug info")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    height, width = frame.shape[:2]
    line_x = int(width * LINE_POSITION_RATIO)  # Calculate line x coordinate

    # Shrink image before findContours
    right_half = frame[:, line_x:].copy()
    small = cv2.resize(right_half, (right_half.shape[1]//2, right_half.shape[0]//2))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_rect = None
    best_approx = None
    for cnt in contours:
        cnt = cnt * 2  # Scale coordinates back up
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h) if h != 0 else 0
                if 0.3 < aspect_ratio < 3.0 and area > max_area:
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

        # Check if rectangle crosses middle line (from right to left)
        rect_center_x_in_full_frame = x + line_x + w/2  # Rectangle center x coordinate in full frame
        
        # Logic to detect crossing middle line
        if last_rect_x is not None:
            # If rectangle moves from right side (x > line_x) to left side (x <= line_x), it crosses the middle line
            if last_rect_x > line_x and rect_center_x_in_full_frame <= line_x:
                rect_crossed_line = True
                text_counter = Counter()
                text_conf_dict.clear()
                # Reset current session position check state
                current_session_position_checked = False
                current_session_position_status = 1
                current_session_warped_img = None
                out_of_area = False
        
        # Update previous frame rectangle position
        last_rect_x = rect_center_x_in_full_frame

        if inside_area:
            if out_of_area and not rect_crossed_line:
                # Only reset when re-entering detection area in non-crossing cases
                text_counter = Counter()
                text_conf_dict.clear()
                # Reset current session position check state
                current_session_position_checked = False
                current_session_position_status = 1
                current_session_warped_img = None
            out_of_area = False
            rect_crossed_line = False  # Reset crossing flag
        else:
            # Still in frame but outside defect area, don't reset, wait for disappearance
            pass

        warped = None
        
        if best_approx is not None:
            warped = four_point_transform(right_half, best_approx)
            if warped is None:
                # Ensure crop range is within image boundaries
                y_start = max(0, y)
                y_end = min(right_half.shape[0], y + h)
                x_start = max(0, x)
                x_end = min(right_half.shape[1], x + w)
                warped = right_half[y_start:y_end, x_start:x_end].copy()
        else:
            # Ensure crop range is within image boundaries
            y_start = max(0, y)
            y_end = min(right_half.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(right_half.shape[1], x + w)
            warped = right_half[y_start:y_end, x_start:x_end].copy()
        
        # Check if warped image is valid
        if warped is not None and warped.size > 0:
            #print(f"[WARP] Warped image valid, size: {warped.shape}")
            pass
        else:
            print("[WARP] Warped image invalid, skipping OCR")
            if warped is not None:
                print(f"   Image size: {warped.size}")
            warped = None

        # Only perform OCR and display when warped image is valid
        if warped is not None and frame_count % OCR_INTERVAL == 0:
           current_ocr_texts = run_ocr_and_collect_text(warped, conf_threshold)
        else:
            current_ocr_texts = {}
        
        # Check if position detection is needed (only on first text detection with reference available)
        if (not current_session_position_checked and 
            current_ocr_texts and 
            reference_warped_img is not None and 
            reference_text is not None):
        

            # Get first detected text for position check
            first_detected_text = list(current_ocr_texts.keys())[0]
            #print(f"[POS] First detected text '{first_detected_text}', performing position check...")
            
            iou_value, is_aligned = calculate_iou(
                warped, 
                first_detected_text, 
                threshold=0.6
            )
            
            current_session_position_status = 1  if is_aligned else 5   # 1: "clear", 5 : "misaligned"
            current_session_position_checked = True
            current_session_warped_img = warped.copy()  # Save this frame's image
            
            #print(f"[POS] Position check complete: IoU={iou_value:.3f}, Status={current_session_position_status}")
            #print("[POS] This session will not repeat position checks")

        # Show ROI window (only when warped is valid)
        if warped is not None and warped.size > 0:
            roi_display = warped.copy()
            if current_ocr_texts:
                # Show cleaned text (current_ocr_texts.keys() are already cleaned)
                cleaned_texts = list(current_ocr_texts.keys())
                text_line = " ".join(cleaned_texts)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(text_line, font, font_scale, font_thickness)
                padding = 10
                cv2.rectangle(roi_display, (0, 0), (roi_display.shape[1], text_height + padding * 2), (255, 255, 255), -1)
                cv2.rectangle(roi_display, (0, 0), (roi_display.shape[1], text_height + padding * 2), (0, 0, 0), 2)
                cv2.putText(roi_display, text_line, (padding, text_height + padding),
                            font, font_scale, (0, 0, 0), font_thickness)
            
            # Save ROI image with OCR text overlay (standardized)
            last_roi_image = roi_display.copy()
            cv2.imshow("ROI + OCR Compare", roi_display)

        if best_approx is not None:
            pts_int = best_approx.astype(int)
            pts_int[:, 0] += line_x  # Adjust coordinate offset
            cv2.polylines(frame, [pts_int], True, (255, 0, 0), 2)
        else:
            cv2.rectangle(frame, (x + line_x, y), (x + line_x + w, y + h), (255, 0, 0), 2)

        top3 = text_counter.most_common(3)
        y0 = 30
        for i, (word, count) in enumerate(top3):
            y = y0 + i * 30
            cv2.putText(frame, f"{word}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        # No rectangle found -> considered as leaving
        last_rect_x = None  # Reset position tracking
        
        if not out_of_area:
            # Output JSON before leaving
            if text_counter:
                top1, count = text_counter.most_common(1)[0]
                confidence = text_conf_dict.get(top1, 0.0)
                
                # Use already checked position status
                position_status = current_session_position_status
                
                # Convert last ROI image to base64 (prioritize image saved during position check)
                image_base64 = ""
                roi_image_to_save = current_session_warped_img if current_session_warped_img is not None else last_roi_image
                image_filename = ""
                if roi_image_to_save is not None:
                    #image_base64 = image_to_base64(frame)
                    image_filename = f"{serial_number}.jpg"
                    cv2.imwrite(r".\analysis\temp\{0}".format(image_filename), frame)
                    image_base64 = image_to_base64(roi_image_to_save)

                result_json = {
                        #"Id" : serial_number,                           #int
                        "Value": top1,                                  #nvarchar(300)
                        "Confidence": float(round(confidence, 3)),      #float
                        "Status": str(position_status),                 #nvarchar(50)
                        "Image_base64": image_base64,                   #nvarchar(MAX) 
                        #"Timestamp": datetime.now().isoformat(),        #datetime
                        "Spacing" :  f"{iou_value:.3f}",                 #nvarchar(50)
                        "Filename" : image_filename
                }
                print("[RESULT] JSON result:Value={0},Confidence={1},Status={2},Image_base64={3}...,Spacing={4},Filename={5}".format(
                    result_json['Value'],
                    result_json['Confidence'],
                    result_json['Status'],
                    result_json['Image_base64'][:5],
                    result_json['Spacing'],
                    result_json['Filename']
                    ) )

                # Insert to database
                print(db_connection)
                if db_connection:
                    insert_to_database(db_connection, result_json)
                else:
                    print("SQL connected fall!!")
                serial_number += 1

            out_of_area = True

    # Draw detection area frame lines
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)

    # DEBUG: Add debug info display
    debug_info = [
        f"Frame: {frame_count}",
        f"FPS: {fps:.2f}",
        #f"Debug: {debug_image_counter} saved",
        f"Status: {current_session_position_status}",
        #f"Checked: {'Yes' if current_session_position_checked else 'No'}"
    ]
    
    for i, info in enumerate(debug_info):
        y = 30 + i * 25
        cv2.putText(frame, info, (10, y + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("a"):  # Move line left
        LINE_POSITION_RATIO = max(0.1, LINE_POSITION_RATIO - 0.05)
        #print(f"[CONTROL] Line position adjusted to: {LINE_POSITION_RATIO:.2f} (left)")
    elif key == ord("d"):  # Move line right
        LINE_POSITION_RATIO = min(0.9, LINE_POSITION_RATIO + 0.05)
        #print(f"[CONTROL] Line position adjusted to: {LINE_POSITION_RATIO:.2f} (right)")
    elif key == ord("s"):  # Force save debug info
        if last_roi_image is not None and reference_warped_img is not None:
            #print("[DEBUG] Manual debug save triggered...")
            # Simulate OCR to get bbox info
            if text_counter:
                current_text = text_counter.most_common(1)[0][0]
                calculate_iou(last_roi_image, current_text, threshold=0.6)
    elif key == ord("r"):  # Reset debug counter
        debug_image_counter = 0
        #print("[DEBUG] Debug counter reset")

cap.release()
cv2.destroyAllWindows()

#print(f"[FINISH] Processing complete, saved {debug_image_counter} debug images")
