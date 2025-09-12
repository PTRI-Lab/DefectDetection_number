import cv2
import numpy as np
import easyocr
import re

def clean_text(raw_text):
    corrected = raw_text.replace('>', '7').replace('?', '7').replace('O', '0')
    return re.sub(r'[^0-9]', '', corrected)

# --- Step 1: Manually select the expected valid region ---
drawing = False
ix, iy = -1, -1
rect = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_img = img.copy()
        cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 255), 2)
        cv2.imshow('Select Expected Region', temp_img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = [(min(ix, x), min(iy, y)), (max(ix, x), max(iy, y))]
        cv2.rectangle(img, rect[0], rect[1], (0, 255, 255), 2)
        cv2.imshow('Select Expected Region', img)
        print(f"Expected region set: {rect}")

# Load image
image_path = './test.jpg'
img = cv2.imread(image_path)
clone = img.copy()
cv2.namedWindow('Select Expected Region')
cv2.setMouseCallback('Select Expected Region', draw_rectangle)

print("Please use the mouse to draw the expected valid region. Press any key to continue when done.")
cv2.imshow('Select Expected Region', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Step 2: OCR + Check if the detected number is inside the valid region ---
def is_inside(bbox, region):
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    x_center = sum(xs) / 4
    y_center = sum(ys) / 4
    (x_min, y_min), (x_max, y_max) = region
    return x_min <= x_center <= x_max and y_min <= y_center <= y_max

reader = easyocr.Reader(['en'])
results = reader.readtext(image_path)

for (bbox, text, conf) in results:
    bbox_int = [tuple(map(int, pt)) for pt in bbox]
    if is_inside(bbox_int, rect):
        color = (0, 255, 0)
        label = 'OK'
        text = clean_text(text)
        print("recogniton result : ",text)
    else:
        color = (0, 0, 255)
        label = 'Defect'
    
    cv2.polylines(clone, [np.array(bbox_int)], isClosed=True, color=color, thickness=2)
    cv2.putText(clone, f'{text} ({label})', bbox_int[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# --- Step 3: Output result ---
cv2.imwrite('output_defect_check.jpg', clone)
print("Defect check completed. Output saved as output_defect_check.jpg")

#python -m venv venv
#.\venv\Scripts\activate.bat
#python.exe -m pip install --upgrade pip
#pip install easyocr 
#pip install git+https://github.com/JaidedAI/EasyOCR.git