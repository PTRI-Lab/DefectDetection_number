import cv2
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_corner_detection():
    """示範如何找到文檔的4個角點"""
    
    # 1. 讀取原始圖像
    image = cv2.imread('../test.jpg')
    original = image.copy()
    
    # 2. 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("步驟1: 轉換為灰度圖")
    cv2.imwrite('step1_gray.jpg', gray)
    
    # 3. 高斯模糊去除噪聲
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    print("步驟2: 高斯模糊去噪")
    cv2.imwrite('step2_blurred.jpg', blurred)
    
    # 4. Canny邊緣檢測
    edges = cv2.Canny(blurred, 50, 150)
    print("步驟3: Canny邊緣檢測")
    cv2.imwrite('step3_edges.jpg', edges)
    print("  → Canny會找到所有亮度變化劇烈的地方")
    print("  → 參數50, 150是低閾值和高閾值")
    
    # 5. 形態學操作連接斷裂的邊緣
    kernel = np.ones((7,7), np.uint8)
    edges_morphed = cv2.dilate(edges, kernel, iterations=2)
    edges_morphed = cv2.erode(edges_morphed, kernel, iterations=2)
    print("步驟4: 形態學操作連接邊緣")
    cv2.imwrite('step4_morphed.jpg', edges_morphed)
    print("  → dilate膨脹：讓白色區域變大，連接斷裂的邊緣")
    print("  → erode侵蝕：讓邊緣回到原來粗細")
    
    # 6. 找輪廓
    contours, _ = cv2.findContours(edges_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"步驟5: 找到 {len(contours)} 個輪廓")
    
    # 7. 按面積排序，找最大的幾個
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # 繪製所有輪廓
    contour_img = original.copy()
    cv2.drawContours(contour_img, contours[:5], -1, (0, 255, 0), 3)
    cv2.imwrite('step5_contours.jpg', contour_img)
    print("  → 綠線顯示找到的前5大輪廓")
    
    # 8. 尋找四邊形
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 30000:  # 太小的輪廓跳過
            continue
        
        print(f"\n檢查輪廓 {i}: 面積 = {area}")
        
        # 計算輪廓周長
        perimeter = cv2.arcLength(contour, True)
        print(f"  周長: {perimeter}")
        
        # 多邊形近似 - 關鍵步驟！
        epsilon = 0.015 * perimeter  # 近似精度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        print(f"  近似後頂點數: {len(approx)}")
        
        # 繪製近似結果
        approx_img = original.copy()
        cv2.drawContours(approx_img, [approx], -1, (0, 0, 255), 5)
        for j, point in enumerate(approx):
            x, y = point[0]
            cv2.circle(approx_img, (x, y), 10, (255, 0, 0), -1)
            cv2.putText(approx_img, str(j), (x+15, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imwrite(f'step6_approx_{i}.jpg', approx_img)
        
        # 如果是四邊形，就是我們要的！
        if len(approx) == 4:
            print(f"  ✓ 找到四邊形！")
            corners = approx.reshape(4, 2)
            print(f"  四個角點座標:")
            for j, (x, y) in enumerate(corners):
                print(f"    點{j}: ({x}, {y})")
            return corners
    
    print("❌ 沒找到四邊形，使用整張圖片")
    h, w = image.shape[:2]
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")

def demonstrate_perspective_transform():
    """示範透視變換原理"""
    
    print("\n" + "="*50)
    print("透視變換 (Perspective Transform) 原理")
    print("="*50)
    
    # 假設我們有4個角點（歪斜的）
    src_points = np.array([
        [100, 50],    # 左上 (歪斜的)
        [400, 80],    # 右上 (歪斜的)  
        [420, 300],   # 右下 (歪斜的)
        [80, 280]     # 左下 (歪斜的)
    ], dtype="float32")
    
    print("原始四邊形座標 (歪斜的):")
    print(f"  左上: {src_points[0]}")
    print(f"  右上: {src_points[1]}")
    print(f"  右下: {src_points[2]}")
    print(f"  左下: {src_points[3]}")
    
    # 計算目標矩形的尺寸
    # 寬度：取右下-左下 和 右上-左上 的最大值
    width1 = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + 
                     ((src_points[2][1] - src_points[3][1]) ** 2))
    width2 = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + 
                     ((src_points[1][1] - src_points[0][1]) ** 2))
    max_width = max(int(width1), int(width2))
    
    # 高度：取右上-右下 和 左上-左下 的最大值
    height1 = np.sqrt(((src_points[1][0] - src_points[2][0]) ** 2) + 
                      ((src_points[1][1] - src_points[2][1]) ** 2))
    height2 = np.sqrt(((src_points[0][0] - src_points[3][0]) ** 2) + 
                      ((src_points[0][1] - src_points[3][1]) ** 2))
    max_height = max(int(height1), int(height2))
    
    print(f"\n計算得到的矩形尺寸: {max_width} x {max_height}")
    
    # 目標矩形座標（完美的矩形）
    dst_points = np.array([
        [0, 0],                           # 左上
        [max_width - 1, 0],               # 右上
        [max_width - 1, max_height - 1],  # 右下
        [0, max_height - 1]               # 左下
    ], dtype="float32")
    
    print("目標矩形座標 (標準矩形):")
    print(f"  左上: {dst_points[0]}")
    print(f"  右上: {dst_points[1]}")
    print(f"  右下: {dst_points[2]}")
    print(f"  左下: {dst_points[3]}")
    
    # 計算透視變換矩陣 - 這是關鍵！
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    print(f"\n透視變換矩陣 (3x3):")
    print(transform_matrix)
    print("這個矩陣定義了如何將歪斜的四邊形映射到標準矩形")
    
    # 示範如何使用變換矩陣
    print("\n變換矩陣的作用：")
    print("對於原圖中的每個點 (x,y)，通過矩陣運算得到新位置 (x',y')")
    print("數學公式：[x' y' w'] = [x y 1] × M")
    print("然後 x'=x'/w', y'=y'/w' 得到最終座標")

def demonstrate_color_detection():
    """示範顏色檢測和二值化"""
    
    print("\n" + "="*50)
    print("顏色檢測與二值化")
    print("="*50)
    
    # 讀取圖像
    image = cv2.imread('./test.jpg')
    
    # 1. HSV色彩空間轉換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    print("步驟1: 轉換到HSV色彩空間")
    print("  → H(色相): 顏色類型 (0-179)")
    print("  → S(飽和度): 顏色純度 (0-255)")  
    print("  → V(明度): 亮度 (0-255)")
    
    # 2. 定義黃色範圍
    lower_yellow = np.array([15, 50, 50])   # HSV下限
    upper_yellow = np.array([35, 255, 255]) # HSV上限
    print(f"\n黃色HSV範圍:")
    print(f"  下限: H={lower_yellow[0]}, S={lower_yellow[1]}, V={lower_yellow[2]}")
    print(f"  上限: H={upper_yellow[0]}, S={upper_yellow[1]}, V={upper_yellow[2]}")
    
    # 3. 創建黃色遮罩
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    print("\n步驟2: 創建黃色遮罩")
    print("  → inRange函數會檢查每個像素是否在指定範圍內")
    print("  → 在範圍內=255(白色), 不在範圍內=0(黑色)")
    cv2.imwrite('demo_yellow_mask.jpg', yellow_mask)
    
    # 4. LAB色彩空間補充檢測
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    print("\n步驟3: LAB色彩空間補充檢測")
    print("  → L: 亮度")
    print("  → A: 綠-紅軸")
    print("  → B: 藍-黃軸 (黃色在這個軸上有高值)")
    
    # B通道閾值化
    _, yellow_mask_lab = cv2.threshold(b, 135, 255, cv2.THRESH_BINARY)
    cv2.imwrite('demo_yellow_mask_lab.jpg', yellow_mask_lab)
    
    # 5. 結合兩種遮罩
    combined_mask = cv2.bitwise_or(yellow_mask, yellow_mask_lab)
    cv2.imwrite('demo_combined_mask.jpg', combined_mask)
    print("步驟4: 結合HSV和LAB遮罩")
    
    # 6. 形態學清理
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite('demo_cleaned_mask.jpg', cleaned_mask)
    print("步驟5: 形態學操作清理遮罩")
    print("  → MORPH_CLOSE: 填補小洞")
    print("  → MORPH_OPEN: 去除小噪點")
    
    # 7. 在原圖上顯示檢測結果
    result = image.copy()
    result[cleaned_mask > 0] = [0, 255, 255]  # 黃色區域標記為青色
    cv2.imwrite('demo_color_detection_result.jpg', result)
    
    return cleaned_mask

def demonstrate_ocr_process():
    """示範OCR處理過程"""
    
    print("\n" + "="*50)
    print("OCR 處理流程")
    print("="*50)
    
    # EasyOCR的工作原理
    print("EasyOCR 工作步驟：")
    print("1. 文字檢測 (Text Detection)")
    print("   → 使用深度學習模型找到圖片中的文字區域")
    print("   → 輸出：每個文字區域的邊界框座標")
    
    print("\n2. 文字識別 (Text Recognition)")  
    print("   → 對每個檢測到的區域進行字符識別")
    print("   → 輸出：文字內容 + 信心度分數")
    
    print("\n3. 後處理")
    print("   → 過濾低信心度結果")
    print("   → 清理和校正文字")
    
    # 信心度解釋
    print("\nEasyOCR 返回格式：")
    print("[(bbox, text, confidence), ...]")
    print("其中：")
    print("  → bbox: 文字區域的4個角點座標")
    print("  → text: 識別出的文字內容") 
    print("  → confidence: 信心度 (0.0-1.0)")
    print("    - 0.9+: 非常確定")
    print("    - 0.7-0.9: 比較確定") 
    print("    - 0.5-0.7: 一般確定")
    print("    - 0.5-: 不太確定")
    
    # 綠色框和信心值說明
    print("\n關於綠色框和信心值：")
    print("❌ 這些不是EasyOCR內建的視覺化功能")
    print("✓ 這些是我們自己用OpenCV繪製的")
    print("  → cv2.polylines(): 繪製邊界框")
    print("  → cv2.putText(): 添加文字標籤")
    print("  → cv2.rectangle(): 繪製矩形框")
    
    # 處理流程說明
    print("\n我們程式的完整流程：")
    print("1. 找到文檔4個角點 → 透視校正")
    print("2. 檢測黃色區域 → 創建遮罩")
    print("3. 在黃色區域中找數字候選區域")
    print("4. 對每個候選區域：")
    print("   a. 多種預處理 (放大、二值化、反轉等)")
    print("   b. 用EasyOCR識別")
    print("   c. 清理文字、計算信心度")
    print("5. 選擇最佳結果並視覺化")

def main():
    """主函數：執行所有示範"""
    
    print("電腦視覺技術詳解")
    print("="*60)
    
    try:
        # 1. 角點檢測示範
        print("1. 文檔角點檢測")
        corners = demonstrate_corner_detection()
        
        # 2. 透視變換示範  
        demonstrate_perspective_transform()
        
        # 3. 顏色檢測示範
        print("3. 顏色檢測")
        mask = demonstrate_color_detection()
        
        # 4. OCR流程說明
        demonstrate_ocr_process()
        
        print("\n" + "="*60)
        print("所有示範完成！")
        print("請查看生成的圖片文件瞭解每個步驟的視覺效果")
        print("="*60)
        
    except Exception as e:
        print(f"示範過程中出現錯誤: {e}")
        print("請確保 test.jpg 文件存在")

if __name__ == "__main__":
    main()