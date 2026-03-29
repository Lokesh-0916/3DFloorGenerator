import cv2
import numpy as np
import json

from t import detect_walls

def detect_windows_by_shape(image_path):
    """Your confirmed working detection logic"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Assuming detect_walls is available in your script
    thick_mask = detect_walls(image_path) 
    thick_mask_fat = cv2.dilate(thick_mask, np.ones((3,3), np.uint8), iterations=1)
    
    thin_v = cv2.subtract(vertical_lines, thick_mask_fat)
    thin_h = cv2.subtract(horizontal_lines, thick_mask_fat)
    thin_lines = cv2.bitwise_or(thin_v, thin_h)
    
    return cv2.morphologyEx(thin_lines, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

def get_windows_json(image_path):
    # 1. Get the high-quality mask from your shape-detection function
    win_pixels = detect_windows_by_shape(image_path)

    # 2. VECTORIZE (Convert pixels to mathematical lines)
    # minLineLength=15 catches short windows; maxLineGap=10 connects fragments
    lines = cv2.HoughLinesP(win_pixels, 1, np.pi/180, 15, minLineLength=15, maxLineGap=10)
    
    raw_windows = []
    if lines is not None:
        raw_windows = [l[0] for l in lines]

    # 3. MERGE LOGIC (Fixes the "Double Detection" bug)
    merged_windows = []
    while len(raw_windows) > 0:
        l1 = raw_windows.pop(0)
        x1, y1, x2, y2 = l1
        is_h = abs(y1 - y2) < abs(x1 - x2)
        
        # Snap to 90 degrees for clean JSON
        if is_h: y2 = y1
        else: x2 = x1
        
        keep = True
        for i in range(len(merged_windows)):
            mx1, my1, mx2, my2 = merged_windows[i]
            m_is_h = abs(my1 - my2) < abs(mx1 - mx2)
            
            if is_h == m_is_h:
                # Calculate distance between parallel segments
                dist = abs(y1 - my1) if is_h else abs(x1 - mx1)
                
                # If they are very close (less than 10px), they are the same window
                if dist < 10: 
                    # Check for overlap/proximity
                    if is_h:
                        if max(x1, x2) >= min(mx1, mx2) - 10 and min(x1, x2) <= max(mx1, mx2) + 10:
                            merged_windows[i] = [min(x1, x2, mx1, mx2), my1, max(x1, x2, mx1, mx2), my1]
                            keep = False; break
                    else:
                        if max(y1, y2) >= min(my1, my2) - 10 and min(y1, y2) <= max(my1, my2) + 10:
                            merged_windows[i] = [mx1, min(y1, y2, my1, my2), mx1, max(y1, y2, my1, my2)]
                            keep = False; break
        
        if keep:
            merged_windows.append([x1, y1, x2, y2])

    # 4. FORMAT FOR JSON
    windows_json = []
    for idx, win in enumerate(merged_windows):
        x1, y1, x2, y2 = win
        windows_json.append({
            "id": f"window_{idx + 1}",
            "start": {"x": int(x1), "y": int(y1)},
            "end": {"x": int(x2), "y": int(y2)},
            "width": int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
        })

    return windows_json