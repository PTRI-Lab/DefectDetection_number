import cv2
import numpy as np
import easyocr
import re
import time

# --- Function to clean up OCR result text ---
def clean_text(raw_text):
    """Enhanced text cleaning for certificate number OCR errors"""
    corrected = raw_text.upper()  # Convert to uppercase first
    
    # Single character replacements for common OCR errors
    corrected = corrected.replace('O', '0').replace('Q', '0')  # O/Q -> 0
    corrected = corrected.replace('I', '1').replace('L', '1').replace('|', '1')  # I/L/| -> 1
    corrected = corrected.replace('Z', '2')     # Z -> 2
    corrected = corrected.replace('E', '3')     # E -> 3
    corrected = corrected.replace('A', '4')     # A -> 4
    corrected = corrected.replace('S', '5')     # S -> 5
    corrected = corrected.replace('G', '6').replace('C', '6')  # G/C -> 6
    corrected = corrected.replace('T', '7').replace('Y', '7')  # T/Y -> 7
    corrected = corrected.replace('B', '8')     # B -> 8 (after multi-char replacements)
    corrected = corrected.replace('P', '8')     # P -> 8
 
    # Remove any remaining non-digit characters
    result = re.sub(r'[^0-9]', '', corrected)
    
    # Post-processing for known certificate number pattern
    if result.startswith('400000') and len(result) > 10:
        # If too long, likely has extra characters, trim to 10 digits
        result = result[:10]
    
    return result

# --- Enhanced preprocessing for yellow text on blue background ---
def preprocess_roi(roi):
    """Enhanced preprocessing for yellow text on blue background"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define yellow color range for certificate numbers
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create yellow color mask
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Morphological operations to clean the mask
    kernel = np.ones((2,2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Fallback to grayscale thresholding if yellow mask fails
    if cv2.countNonZero(mask) < 50:  # If too few yellow pixels detected
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert mask to 3-channel image
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return result

# --- Global variables for ROI drawing ---
drawing = False
ix, iy = -1, -1
roi_rect = []
current_frame = None  # Store current frame for ROI selection

# --- Mouse callback to select ROI ---
def select_roi(event, x, y, flags, param):
    global drawing, ix, iy, roi_rect, current_frame
    if current_frame is None:
        return
        
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = current_frame.copy()
        cv2.rectangle(temp, (ix, iy), (x, y), (0, 255, 255), 2)
        cv2.imshow("Select ROI", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_rect = [(min(ix, x), min(iy, y)), (max(ix, x), max(iy, y))]
        print(f"ROI selected: {roi_rect}")

# --- Initialize camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# --- Show first frame to select ROI ---
print("Please drag with mouse to select ROI area (recommend selecting number area below the bear)...")
ret, frame = cap.read()
if not ret:
    print("Cannot read frame.")
    cap.release()
    exit()

# Store frame for ROI selection
current_frame = frame.copy()
cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", select_roi)
cv2.imshow("Select ROI", current_frame)
cv2.waitKey(0)
cv2.destroyWindow("Select ROI")

# Verify ROI selection was successful
if not roi_rect:
    print("No ROI selected, exiting...")
    cap.release()
    cv2.destroyAllWindows()
    exit()

print(f"ROI successfully selected: {roi_rect}")

# --- Initialize OCR reader with optimized settings ---
reader = easyocr.Reader(['en'], gpu=False)

# --- Main loop ---
print("Starting OCR recognition. Press 'q' to quit, press 'r' to reselect ROI.")
last_ocr_time = time.time()
ocr_interval = 0.5  # Reduced interval for better responsiveness
last_result = ""
confidence_threshold = 0.2  # Lowered threshold for certificate OCR

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera")
        break
    
    # Verify frame is valid
    if frame is None or frame.size == 0:
        print("Invalid frame received")
        continue

    display_frame = frame.copy()
    
    # Verify display_frame is valid
    if display_frame is None or display_frame.size == 0:
        print("Failed to copy frame")
        continue

    if roi_rect:
        (x1, y1), (x2, y2) = roi_rect
        h, w = frame.shape[:2]
        
        # Enhanced boundary checking
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))  # Ensure x2 > x1
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))  # Ensure y2 > y1

        # Skip if ROI too small or invalid
        if (x2 - x1 < 10) or (y2 - y1 < 10):
            print("ROI area too small, skipping recognition.")
            continue

        # Additional safety check
        if x1 >= x2 or y1 >= y2:
            print("Invalid ROI coordinates, skipping.")
            continue

        try:
            roi = frame[y1:y2, x1:x2]
            
            # Check if ROI is valid
            if roi.size == 0:
                print("Empty ROI, skipping.")
                continue
                
            # --- Enhanced preprocessing for yellow text ---
            processed_roi = preprocess_roi(roi)
            
            # Perform OCR with confidence checking
            current_time = time.time()
            if current_time - last_ocr_time >= ocr_interval:
                try:
                    # Use detail=1 to get confidence scores
                    results = reader.readtext(processed_roi, detail=1, allowlist='0123456789')
                    if results:
                        # Get result with highest confidence
                        best_result = max(results, key=lambda x: x[2])  # x[2] is confidence
                        text, confidence = best_result[1], best_result[2]
                        
                        if confidence > confidence_threshold:
                            cleaned = clean_text(text)
                            print(f"Raw OCR: '{text}' -> Cleaned: '{cleaned}' (confidence: {confidence:.2f})")
                            if len(cleaned) >= 10 and len(cleaned) <= 12:  # Certificate numbers are typically 10 digits
                                last_result = cleaned
                                print(f"✓ OCR result accepted: {cleaned}")
                            elif len(cleaned) >= 8:  # Accept shorter results with warning
                                last_result = cleaned
                                print(f"⚠ OCR result (short): {cleaned}")
                            else:
                                print(f"✗ Detected text too short: {cleaned} (len: {len(cleaned)})")
                        else:
                            # Show what was detected even if confidence is low
                            cleaned = clean_text(text)
                            print(f"Low confidence - Raw: '{text}' -> Cleaned: '{cleaned}' (confidence: {confidence:.2f})")
                    else:
                        last_result = "No text detected."
                        print("No text detected.")
                except Exception as e:
                    print("OCR Error:", e)
                    last_result = ""
                last_ocr_time = current_time

            # Draw ROI rectangle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Draw result text above the ROI with better positioning
            if last_result:
                # Calculate text position to ensure it stays within frame
                text_size = cv2.getTextSize(last_result, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                text_x = max(10, x1 + (x2 - x1 - text_size[0]) // 2)
                text_y = max(40, y1 - 15) if y1 > 50 else y2 + 40
                
                # Add text background
                cv2.rectangle(display_frame, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(display_frame, last_result, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

            # Display processed ROI in small window
            if processed_roi is not None and processed_roi.size > 0:
                try:
                    roi_display = cv2.resize(processed_roi, (200, 100))
                    display_frame[10:110, 10:210] = roi_display
                except Exception as e:
                    print(f"Error displaying ROI preview: {e}")
                    
        except Exception as e:
            print(f"Error processing ROI: {e}")
            continue

    # Show result in real-time window
    try:
        # Debug: Check display_frame status
        if display_frame is None:
            print("Warning: display_frame is None")
            continue
        if display_frame.size == 0:
            print("Warning: display_frame is empty")
            continue
            
        cv2.imshow("Live OCR - Press 'q' to quit, 'r' to reselect ROI", display_frame)
    except Exception as e:
        print(f"Error displaying frame: {e}")
        continue

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting program...")
        break
    elif key == ord('r'):
        # Reselect ROI safely
        print("Reselecting ROI...")
        try:
            # Get fresh frame for ROI selection
            ret, fresh_frame = cap.read()
            if not ret or fresh_frame is None:
                print("Failed to get fresh frame for ROI selection")
                continue
                
            cv2.destroyWindow("Live OCR - Press 'q' to quit, 'r' to reselect ROI")
            roi_rect = []  # Reset ROI
            current_frame = fresh_frame.copy()  # Update current frame for selection
            cv2.namedWindow("Select ROI")
            cv2.setMouseCallback("Select ROI", select_roi)
            cv2.imshow("Select ROI", current_frame)
            cv2.waitKey(0)
            cv2.destroyWindow("Select ROI")
            
            # Verify new ROI was selected
            if not roi_rect:
                print("No ROI selected during reselection")
            else:
                print(f"New ROI selected: {roi_rect}")
                
        except Exception as e:
            print(f"Error during ROI reselection: {e}")

cap.release()
cv2.destroyAllWindows()
