import os, time, csv, json, glob
import cv2
import numpy as np
import pyodbc

INPUT_DIR = "temp"
OUTPUT_DIR = "temp_out"
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
CSV_PATH = os.path.join(OUTPUT_DIR, "alignment_results.csv")
BASELINE_PATH = os.path.join(OUTPUT_DIR, "_baseline.json")

GREEN_STRICT = ((40, 60, 40), (85, 255, 255))
GREEN_RELAX  = ((30, 40, 30), (95, 255, 255))

def _log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] "), msg)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def four_point_transform_with_margin(image, pts, margin=30):
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    if margin > 0:
        out = np.zeros((maxHeight + 2*margin, maxWidth + 2*margin, 3), dtype=warped.dtype)
        out[margin:margin+maxHeight, margin:margin+maxWidth] = warped
        return out
    return warped



def find_quad(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            return approx.reshape(4, 2).astype("float32")
    return None

def enforce_landscape(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if img.shape[0] > img.shape[1] else img

def rectify_one(path, margin=30):
    img = cv2.imread(path)
    if img is None:
        print(f"找不到檔案: {path}"); return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    quad = find_quad(gray)
    if quad is None:
        print(f"[保底] 使用 minAreaRect：{path}")
        cnts, _ = cv2.findContours(cv2.Canny(gray, 10, 60), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype("float32")
        warped = four_point_transform_with_margin(img, box, margin=margin)
    else:
        warped = four_point_transform_with_margin(img, quad, margin=margin)

    return enforce_landscape(warped)


def _largest_rect(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return c, (x, y, w, h)

def _save_debug(img, path):
    try:
        cv2.imwrite(path, img)
    except Exception:
        pass




def _detect_frame_rect(img_bgr, hsv, debug_prefix=None):
    k = np.ones((7,7), np.uint8)

    lower, upper = np.array(GREEN_STRICT[0]), np.array(GREEN_STRICT[1])
    mask1 = cv2.morphologyEx(cv2.inRange(hsv, lower, upper), cv2.MORPH_CLOSE, k, iterations=1)
    _, rect = _largest_rect(mask1)
    if rect is not None and rect[2]*rect[3] > 1000:
        if debug_prefix: _save_debug(mask1, os.path.join(DEBUG_DIR, f"{debug_prefix}_frame_strict.png"))
        return rect

    lower, upper = np.array(GREEN_RELAX[0]), np.array(GREEN_RELAX[1])
    mask2 = cv2.morphologyEx(cv2.inRange(hsv, lower, upper), cv2.MORPH_CLOSE, k, iterations=1)
    _, rect = _largest_rect(mask2)
    if rect is not None and rect[2]*rect[3] > 1000:
        if debug_prefix: _save_debug(mask2, os.path.join(DEBUG_DIR, f"{debug_prefix}_frame_relax.png"))
        return rect

    # grayscale fallback
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k, iterations=2)
    _, rect = _largest_rect(thr)
    if rect is not None and rect[2]*rect[3] > 1000:
        if debug_prefix: _save_debug(thr, os.path.join(DEBUG_DIR, f"{debug_prefix}_frame_fallback.png"))
        return rect
    return None




def _shrink_rect(x,y,w,h, inset):
    nx = x + inset
    ny = y + inset
    nw = max(1, w - 2*inset)
    nh = max(1, h - 2*inset)
    return nx, ny, nw, nh

def _detect_panel_inside_frame(img_bgr, frame_rect, debug_prefix=None):
    x,y,w,h = frame_rect
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    V = hsv[:,:,2]

    # Otsu bright vs dark, we want dark
    _, otsu_bin = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # bright=255
    dark1 = cv2.bitwise_not(otsu_bin)
    # Absolute low-V threshold as backup
    dark2 = cv2.inRange(V, 0, min(90, int(_+20)))
    mask = cv2.bitwise_or(dark1, dark2)

    k = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    # pull away from bright border
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)

    _, rect = _largest_rect(mask)
    if rect is None:
        return None

    px,py,pw,ph = rect

    # enforce: panel must be inside frame with margins
    if not (px>2 and py>2 and px+pw < w-2 and py+ph < h-2):
        # if touching borders too much, shrink heuristically
        px,py,pw,ph = max(2,px), max(2,py), min(w-4, pw), min(h-4, ph)

    # shrink panel to avoid border bleed
    inset = max(4, int(0.04*min(pw,ph)))
    sx,sy,sw,sh = _shrink_rect(px,py,pw,ph, inset)

    if debug_prefix:
        _save_debug(mask, os.path.join(DEBUG_DIR, f"{debug_prefix}_panel_mask.png"))
    return (x+sx, y+sy, sw, sh)

def _tighten_proj(bin_img, axis=0, min_frac=0.05):
    counts = (bin_img > 0).sum(axis=axis)
    length = bin_img.shape[1-axis]
    thresh = max(1, int(round(min_frac * length)))
    idx = np.where(counts >= thresh)[0]
    if idx.size == 0:
        return 0, bin_img.shape[1] if axis==0 else bin_img.shape[0]
    start, end = int(idx[0]), int(idx[-1])
    return start, end+1

def _detect_text_in_panel(img_bgr, panel_rect, debug_prefix=None):
    x,y,w,h = panel_rect
    roi = img_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_c = clahe.apply(gray)

    bin_ad = cv2.adaptiveThreshold(gray_c, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, -5)
    _, bin_ot = cv2.threshold(gray_c, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bin_union = cv2.bitwise_or(bin_ad, bin_ot)

    # Clean noise
    bin_clean = cv2.morphologyEx(bin_union, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), iterations=1)

    # Connect characters horizontally
    k_w = max(7, int(0.18 * h))
    dil = cv2.dilate(bin_clean, cv2.getStructuringElement(cv2.MORPH_RECT,(k_w,3)), iterations=1)

    # find multiple candidates
    cnts, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        if debug_prefix:
            _save_debug(bin_clean, os.path.join(DEBUG_DIR, f"{debug_prefix}_text_bin.png"))
            _save_debug(dil,       os.path.join(DEBUG_DIR, f"{debug_prefix}_text_dil.png"))
        return None, None

    panel_cy = h/2.0
    best_score, best_rect = -1, None
    for c in cnts:
        x0,y0,w0,h0 = cv2.boundingRect(c)
        area = w0*h0
        hf = h0/float(h)
        wf = w0/float(w)
        ar = w0/float(max(1,h0))

        # hard constraints
        if area < 0.01*w*h or area > 0.40*w*h:   # 不要太小/太大
            continue
        if not (0.10 <= hf <= 0.40):             # 文字高度相對面板
            continue
        if wf < 0.25:                            # 需要夠寬
            continue
        if ar < 2.0:                             # 長寬比需大於 2
            continue

        # center preference
        cy = y0 + h0/2.0
        center_score = 1.0 - min(1.0, abs(cy - panel_cy)/(h/2.0))
        width_score  = min(1.0, wf/0.8)          # 越寬越好，0.8 封頂
        height_score = 1.0 - abs(hf - 0.22)/0.22 # 偏好 ~22% 高
        score = 0.5*center_score + 0.35*width_score + 0.15*max(0.0,height_score)
        if score > best_score:
            best_score, best_rect = score, (x0,y0,w0,h0)

    if best_rect is None:
        c = max(cnts, key=cv2.contourArea)
        best_rect = cv2.boundingRect(c)

    tx,ty,tw,th = best_rect

    # tighten via projections
    y0,y1 = _tighten_proj(bin_clean[ty:ty+th, tx:tx+tw], axis=1, min_frac=0.06)
    ty2,th2 = ty+y0, max(1, y1-y0)
    x0,x1 = _tighten_proj(dil[ty2:ty2+th2, tx:tx+tw], axis=0, min_frac=0.06)
    tx2,tw2 = tx+x0, max(1, x1-x0)

    # clamp to panel
    fx = x + max(0, min(tx2, w-1))
    fy = y + max(0, min(ty2, h-1))
    fw = min(w - (fx - x), tw2)
    fh = min(h - (fy - y), th2)

    if debug_prefix:
        _save_debug(bin_clean, os.path.join(DEBUG_DIR, f"{debug_prefix}_text_bin.png"))
        _save_debug(dil,       os.path.join(DEBUG_DIR, f"{debug_prefix}_text_dil.png"))

    return (fx, fy, fw, fh), dil


def detect_rois(img_bgr, debug_prefix=None):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    frame_rect = _detect_frame_rect(img_bgr, hsv, debug_prefix=debug_prefix)
    if frame_rect is None:
        return None

    panel_rect = _detect_panel_inside_frame(img_bgr, frame_rect, debug_prefix=debug_prefix)
    if panel_rect is None:
        return None

    text_rect, _ = _detect_text_in_panel(img_bgr, panel_rect, debug_prefix=debug_prefix)
    if text_rect is None:
        return None

    x_p, y_p, w_p, h_p = frame_rect
    x_t, y_t, w_t, h_t = text_rect

    top_gap    = y_t - y_p
    left_gap   = x_t - x_p
    right_gap  = (x_p + w_p) - (x_t + w_t)
    bottom_gap = (y_p + h_p) - (y_t + h_t)

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return {
        "rgb": rgb,
        "x_p": x_p, "y_p": y_p, "w_p": w_p, "h_p": h_p,
        "x_t": x_t, "y_t": y_t, "w_t": w_t, "h_t": h_t,
        "top_gap": top_gap, "left_gap": left_gap,
        "right_gap": right_gap, "bottom_gap": bottom_gap
    }


def annotate_and_save(result, out_path):
    rgb = result["rgb"].copy()
    x_p,y_p,w_p,h_p = result["x_p"], result["y_p"], result["w_p"], result["h_p"]
    x_t,y_t,w_t,h_t = result["x_t"], result["y_t"], result["w_t"], result["h_t"]
    top_gap,left_gap = result["top_gap"], result["left_gap"]
    right_gap,bottom_gap = result["right_gap"], result["bottom_gap"]

    cv2.rectangle(rgb, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 255, 0), 3)
    cv2.rectangle(rgb, (x_t, y_t), (x_t + w_t, y_t + h_t), (255, 0, 0), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb, f"top: {top_gap}", (x_t, y_t - 20), font, 0.6, (0, 0, 0), 2)
    cv2.putText(rgb, f"left: {left_gap}", (x_t - 120, y_t + int(h_t / 2)), font, 0.6, (0, 0, 0), 2)
    cv2.putText(rgb, f"right: {right_gap}", (x_t + w_t + 10, y_t + int(h_t / 2)), font, 0.6, (0, 0, 0), 2)
    cv2.putText(rgb, f"bottom: {bottom_gap}", (x_t, y_t + h_t + 30), font, 0.6, (0, 0, 0), 2)

    cv2.putText(rgb, f"Text ROI: {w_t}x{h_t}", (x_t, y_t - 40), font, 0.6, (255, 0, 0), 2)
    cv2.putText(rgb, f"Green ROI: {w_p}x{h_p}", (x_p, y_p - 10), font, 0.6, (0, 128, 0), 2)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)

def compute_baseline(result):
    h_p = result["h_p"]; w_p = result["w_p"]
    h_t = result["h_t"]
    top_gap = result["top_gap"]
    left_gap = result["left_gap"]

    top_ratio = top_gap / h_p if h_p != 0 else 0.0
    left_ratio = left_gap / w_p if w_p != 0 else 0.0
    no_rotate_h = h_t
    return top_ratio, left_ratio, no_rotate_h

def analyze_against_baseline(result, baseline):
    h_p = result["h_p"]; w_p = result["w_p"]
    h_t = result["h_t"]
    top_gap = result["top_gap"]; left_gap = result["left_gap"]

    base_top_ratio = baseline["top_ratio"]
    base_left_ratio = baseline["left_ratio"]
    base_no_rotate_h = baseline["no_rotate_h"]

    expected_top = h_p * base_top_ratio
    expected_left = w_p * base_left_ratio

    delta_v = top_gap - expected_top
    v_shift = 0 if abs(delta_v) < 1e-6 else (int(round(abs(delta_v))) * (1 if delta_v > 0 else -1))

    delta_h = left_gap - expected_left
    h_shift = 0 if abs(delta_h) < 1e-6 else (int(round(abs(delta_h))) * (-1 if delta_h > 0 else 1))

    rotate_flag = 0 if (base_no_rotate_h - 6) <= h_t <= (base_no_rotate_h + 6) else 1
    return v_shift, h_shift, rotate_flag



def wait_until_stable(path, checks=3, interval=0.3, max_tries=60):
    last = -1; stable = 0
    for _ in range(max_tries):
        if not os.path.exists(path):
            time.sleep(interval); continue
        sz = os.path.getsize(path)
        if sz == last and sz > 0:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0; last = sz
        time.sleep(interval)
    return False

def clean_start():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # purge output directory
    for p in glob.glob(os.path.join(OUTPUT_DIR, "*")):
        try:
            if os.path.isfile(p) or os.path.islink(p):
                os.remove(p)
            else:
                import shutil
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            _log(f"[CLEAN] Failed to remove {p}: {e}")
    for p in glob.glob(os.path.join(INPUT_DIR, "*.jpg")) + glob.glob(os.path.join(INPUT_DIR, "*.png")):
        try:
            os.remove(p)
            _log(f"[CLEAN] 移除 {p}")
        except Exception as e:
            _log(f"[CLEAN] 無法刪除 {p}: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "v_shift", "h_shift", "rotate"])

    _log("[CLEAN] temp_out/ cleared and CSV reinitialized.")

def process_image_file(in_path, out_path, margin=30):
    try:
        warped = rectify_one(in_path, margin=margin)
        if warped is None:
            _log(f"[SKIP] 無法處理：{os.path.basename(in_path)}")
            return False
        cv2.imwrite(out_path, warped)
        _log(f"[RECTIFIED] {os.path.basename(in_path)} → {out_path}")
        return True
    except Exception as e:
        _log(f"[ERROR] {os.path.basename(in_path)}：{e}")
        return False

def analyze_output_file(out_path):
    if not wait_until_stable(out_path):
        _log(f"[WAIT] 輸出檔案尚未穩定：{os.path.basename(out_path)}"); return

    img = cv2.imread(out_path)
    if img is None:
        _log(f"[ERROR] 讀取失敗：{out_path}"); return

    fname = os.path.basename(out_path)
    stem = os.path.splitext(fname)[0]

    res = detect_rois(img, debug_prefix=stem)
    if res is None:
        _log(f"[WARN] 找不到指定 ROI：{fname}（已輸出 debug 掩膜於 temp_out/debug/）")
        return

    analysis_img_path = os.path.join(OUTPUT_DIR, f"analysis_{stem}.jpg")
    REFERENCE_PATH = r"../reference3.jpg"

    if not os.path.exists(BASELINE_PATH):  # 如果 baseline.json 還沒建立，就去處理 reference
        ref_img = cv2.imread(REFERENCE_PATH)
        if ref_img is None:
            _log(f"[ERROR] 找不到 reference.jpg：{REFERENCE_PATH}")
            return
        ref_res = detect_rois(ref_img, debug_prefix="reference")
        if ref_res is None:
            _log("[ERROR] 無法從 reference.jpg 偵測 ROI")
            return

        annotate_and_save(ref_res, os.path.join(OUTPUT_DIR, "analysis_reference.jpg"))
        top_ratio, left_ratio, no_rotate_h = compute_baseline(ref_res)
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "top_ratio": float(top_ratio),
                "left_ratio": float(left_ratio),
                "no_rotate_h": int(no_rotate_h)
            }, f)
        _log(f"[BASELINE] 已從 {REFERENCE_PATH} 建立 baseline.")

    baseline = json.load(open(BASELINE_PATH, "r", encoding="utf-8"))
    annotate_and_save(res, analysis_img_path)

    v_shift, h_shift, rotate_flag = analyze_against_baseline(res, baseline)
    
    if db_connection:
        update_shift_data(db_connection, fname, v_shift, h_shift, rotate_flag)
    else:
        print("SQL connected fall!!")
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([fname, v_shift, h_shift, rotate_flag])
        

    _log(f"[ANALYZED] {fname} → v_shift={v_shift}, h_shift={h_shift}, rotate={rotate_flag}; CSV updated.")


def watch_and_process(margin=30, scan_interval=1.5):
    clean_start()
    _log(f"開始監看：{INPUT_DIR} → {OUTPUT_DIR} (margin={margin})")
    processed = set()
    analyzed = set()

    try:
        while True:
            candidates = [f for f in os.listdir(INPUT_DIR)
                          if f.lower().endswith(".jpg") and f[:-4].isdigit()]
            candidates.sort(key=lambda x: int(x[:-4]))

            for fname in candidates:
                in_path = os.path.join(INPUT_DIR, fname)
                out_path = os.path.join(OUTPUT_DIR, fname)

                if fname not in processed:
                    if not wait_until_stable(in_path):
                        _log(f"[WAIT] 輸入檔案尚未穩定：{fname}")
                        continue
                    if process_image_file(in_path, out_path, margin=margin):
                        processed.add(fname)

                if fname in processed and fname not in analyzed:
                    analyze_output_file(out_path)
                    analyzed.add(fname)

            time.sleep(scan_interval)
    except KeyboardInterrupt:
        _log("停止監看（KeyboardInterrupt）。")

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
 
        conn.commit()
        print(f"[DB] Database connected and UV table cleared for testing")
        print(f"[DB] Connected to: {DATABASE_CONFIG['server']}/OCR")
        
        return conn
        
    except Exception as e:
        print(f"[DB ERROR] Failed to connect to database: {e}")
        return None

def update_shift_data(conn, filename, v_shift, h_shift, rotate):
    """
    Update v_shift, h_shift, and rotate columns for a record in UV table based on Filename.
    """
    if conn is None:
        print("[DB ERROR] No database connection available")
        return False

    try:
        cursor = conn.cursor()

        update_sql = """
        UPDATE UV
        SET v_shift = ?, 
            h_shift = ?, 
            rotate = ?
        WHERE Filename = ?
        """

        cursor.execute(update_sql, (v_shift, h_shift, rotate, filename))
        conn.commit()

        if cursor.rowcount == 0:
            print(f"[DB WARNING] No rows updated — check if Filename '{filename}' exists.")
            return False
        else:
            print(f"[DB] Updated record for Filename='{filename}' with v_shift={v_shift}, h_shift={h_shift}, rotate={rotate}")
            return True

    except Exception as e:
        print(f"[DB ERROR] Failed to update UV table: {e}")
        return False


# Initialize database connection at startup
db_connection = init_database()

# Start watching with panel-inset text ROI (v4)
watch_and_process(margin=30, scan_interval=1.5)



