import cv2
import numpy as np
import easyocr
import re
from scipy.spatial import distance as dist

def clean_text(raw_text):
    """清理OCR識別的文字，轉換常見的誤識別字符"""
    corrected = raw_text.replace('>', '7').replace('?', '7').replace('O', '0')
    corrected = corrected.replace('l', '1').replace('I', '1').replace('|', '1')
    corrected = corrected.replace('S', '5').replace('s', '5').replace('G', '6')
    return re.sub(r'[^0-9]', '', corrected)

def order_points(pts):
    """將四個點按照左上、右上、右下、左下的順序排列"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角
    rect[2] = pts[np.argmax(s)]  # 右下角
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角
    rect[3] = pts[np.argmax(diff)]  # 左下角
    
    return rect

def four_point_transform(image, pts):
    """透視變換將傾斜的四邊形轉換為矩形"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped, M

def detect_document(image):
    """檢測文檔的四個角點"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # 更強的形態學操作
    kernel = np.ones((7,7), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=2)
    edged = cv2.erode(edged, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30000:
            continue
            
        epsilon = 0.015 * cv2.arcLength(contour, True)  # 更精確的近似
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2)
    
    h, w = image.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

def enhance_yellow_numbers(image):
    """專門針對黃色數字的增強處理"""
    # 轉換到多個色彩空間進行處理
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 更精確的黃色範圍檢測
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask_hsv = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # LAB空間中的黃色檢測
    l, a, b = cv2.split(lab)
    yellow_mask_lab = cv2.threshold(b, 135, 255, cv2.THRESH_BINARY)[1]
    
    # 結合遮罩
    yellow_mask = cv2.bitwise_or(yellow_mask_hsv, yellow_mask_lab)
    
    # 形態學操作清理遮罩
    kernel = np.ones((3,3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return yellow_mask

def find_number_regions(image, yellow_mask):
    """尋找可能的數字區域"""
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    number_candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # 降低面積閾值
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        
        # 擴大長寬比範圍，因為數字可能比較密集
        if 1.5 < aspect_ratio < 20 and w > 30 and h > 10:
            # 擴展邊界框
            padding = 15
            x = max(0, x - padding)
            y = max(0, y - padding) 
            w = min(image.shape[1] - x, w + 2*padding)
            h = min(image.shape[0] - y, h + 2*padding)
            
            number_candidates.append((x, y, w, h, area))
    
    # 按面積和位置排序，優先選擇較大且位於下方的區域
    if number_candidates:
        number_candidates.sort(key=lambda x: (-x[4], x[1]))  # 面積降序，y座標升序
    
    return number_candidates

def preprocess_for_ocr(roi):
    """針對OCR的多種預處理方法"""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()
    
    # 調整尺寸以提高OCR精度
    scale_factor = 3
    height, width = gray.shape
    resized = cv2.resize(gray, (width * scale_factor, height * scale_factor), 
                        interpolation=cv2.INTER_CUBIC)
    
    processed_versions = []
    
    # 1. 原始放大版本
    processed_versions.append(("resized_original", resized))
    
    # 2. 反轉 + 二值化
    inverted = cv2.bitwise_not(resized)
    _, binary1 = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_versions.append(("inverted_binary", binary1))
    
    # 3. CLAHE增強對比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    _, binary2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_versions.append(("clahe_binary", binary2))
    
    # 4. 自適應閾值
    adaptive = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 15, 5)
    processed_versions.append(("adaptive_thresh", adaptive))
    
    # 5. 形態學處理
    kernel_small = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel_small)
    processed_versions.append(("morphology", morph))
    
    return processed_versions

def process_image(image_path):
    """主要處理函數"""
    original = cv2.imread(image_path)
    if original is None:
        print(f"無法讀取圖像: {image_path}")
        return
    
    print("正在檢測文檔邊界...")
    corners = detect_document(original)
    
    # 繪製檢測到的角點
    debug_img = original.copy()
    for i, point in enumerate(corners):
        cv2.circle(debug_img, tuple(point.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(debug_img, str(i), tuple(point.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    print("正在進行透視校正...")
    warped, _ = four_point_transform(original, corners)
    
    print("正在檢測黃色區域...")
    yellow_mask = enhance_yellow_numbers(warped)
    
    print("正在尋找數字區域...")
    number_regions = find_number_regions(warped, yellow_mask)
    
    # 初始化OCR
    reader = easyocr.Reader(['en'], gpu=False)
    
    all_results = []
    
    # 處理每個可能的數字區域
    for idx, (x, y, w, h, area) in enumerate(number_regions):
        print(f"\n處理區域 {idx + 1}: ({x}, {y}, {w}, {h}), 面積: {area}")
        
        roi = warped[y:y+h, x:x+w]
        cv2.imwrite(fr'.\static\roi_{idx}.jpg', roi)
        
        # 獲取多種預處理版本
        processed_versions = preprocess_for_ocr(roi)
        
        best_result = None
        best_confidence = 0
        
        for method_name, processed_roi in processed_versions:
            print(f"  嘗試方法: {method_name}")
            cv2.imwrite(fr'.\static\roi_{idx}_{method_name}.jpg', processed_roi)
            
            try:
                # OCR識別
                ocr_results = reader.readtext(processed_roi, 
                                            paragraph=False,
                                            width_ths=0.7,
                                            height_ths=0.7)
                
                for (bbox, text, confidence) in ocr_results:
                    print(f"    原始文字: '{text}', 信心度: {confidence:.3f}")
                    
                    if confidence > 0.2:  # 降低信心度閾值
                        cleaned_text = text# clean_text(text)
                        #print(f"    清理後: '{cleaned_text}'")
                        
                        if cleaned_text and len(cleaned_text) >= 4:  # 降低長度要求
                            if confidence > best_confidence:
                                best_result = {
                                    'text': cleaned_text,
                                    'original_text': text,
                                    'confidence': confidence,
                                    'region': (x, y, w, h),
                                    'method': method_name,
                                    'region_index': idx
                                }
                                best_confidence = confidence
                                print(f"    *** 找到更好結果: {cleaned_text} ***")
                
            except Exception as e:
                print(f"    OCR錯誤: {e}")
                continue
        
        if best_result:
            all_results.append(best_result)
    
    # 繪製結果
    result_img = warped.copy()
    
    # 顯示黃色遮罩
    yellow_overlay = result_img.copy()
    yellow_overlay[yellow_mask > 0] = [0, 255, 255]  # 黃色高亮
    result_img = cv2.addWeighted(result_img, 0.7, yellow_overlay, 0.3, 0)
    
    # 繪製所有檢測到的區域
    for idx, (x, y, w, h, area) in enumerate(number_regions):
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(result_img, f'Region {idx+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 繪製識別結果
    for result in all_results:
        x, y, w, h = result['region']
        text = result['text']
        confidence = result['confidence']
        method = result['method']
        
        # 用綠色框標記成功識別的區域
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # 添加識別結果標籤
        label = f'{text} ({confidence:.2f})'
        cv2.putText(result_img, label, (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_img, method, (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存調試圖像
    cv2.imwrite(r'.\static\debug_corners.jpg', debug_img)
    cv2.imwrite(r'.\static\warped_document.jpg', warped)
    cv2.imwrite(r'.\static\yellow_mask.jpg', yellow_mask)
    cv2.imwrite(r'.\static\final_result.jpg', result_img)
    
    # 輸出結果
    print("\n" + "="*50)
    print("檢測結果:")
    print("="*50)
    
    if all_results:
        for i, result in enumerate(all_results, 1):
            print(f"結果 {i}:")
            print(f"  識別數字: {result['text']}")
            print(f"  原始文字: {result['original_text']}")
            print(f"  信心度: {result['confidence']:.3f}")
            print(f"  使用方法: {result['method']}")
            print(f"  區域座標: {result['region']}")
            print("-" * 30)
    else:
        print("未檢測到任何數字")
        print("\n調試建議:")
        print("1. 檢查 yellow_mask.jpg 確認黃色區域是否正確檢測")
        print("2. 檢查 roi_*.jpg 確認數字區域是否正確提取")
        print("3. 檢查各種預處理版本的效果")
    
    print(f"\n保存的調試文件:")
    print("- debug_corners.jpg: 角點檢測結果")
    print("- warped_document.jpg: 透視校正結果") 
    print("- yellow_mask.jpg: 黃色區域遮罩")
    print("- roi_*.jpg: 提取的數字區域")
    print("- roi_*_*.jpg: 各種預處理結果")
    print("- final_result.jpg: 最終標註結果")
    
    return all_results

if __name__ == "__main__":
    image_path = r'.\static\standard.png'
    results = process_image(image_path)
    
    if results:
        print(f"\n✓ 成功識別到 {len(results)} 個數字序列")
        for result in results:
            print(f"  → {result['text']}")
    else:
        print("\n✗ 未能識別到數字序列")
        print("請檢查生成的調試圖像以診斷問題")