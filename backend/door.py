import cv2
import numpy as np

def get_Cordinates(image_path='test/F2.png'):
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- STEP 1: ROBUST TEXT REMOVAL ---
    _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    
    # Remove ultra-thin strokes (Text removal)
    thin_kernel = np.ones((2, 2), np.uint8)
    thin_removed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, thin_kernel)
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thin_removed, connectivity=8)
    no_text = np.zeros_like(thresh)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if h == 0 or w == 0: continue
        
        aspect_ratio = w / float(h)
        
        # 🚫 Window/Text rejection conditions from your working file
        if (area < 800 or (w < 80 and h < 80) or aspect_ratio > 6 or aspect_ratio < 0.2):
            continue
        no_text[labels == i] = 255

    no_text = cv2.medianBlur(no_text, 3)

    # --- STEP 2: WALL REMOVAL ---
    kernel_h = np.ones((25, 5), np.uint8)
    kernel_v = np.ones((5, 25), np.uint8)
    walls_h = cv2.morphologyEx(no_text, cv2.MORPH_OPEN, kernel_h)
    walls_v = cv2.morphologyEx(no_text, cv2.MORPH_OPEN, kernel_v)
    walls = cv2.bitwise_or(walls_h, walls_v)
    no_walls = cv2.subtract(no_text, walls)
    no_walls = cv2.morphologyEx(no_walls, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # --- STEP 3: GATE & WINDOW DETECTION ---
    contours, _ = cv2.findContours(no_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gate_list = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 50 or area < 30: continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        if rect_area == 0: continue

        solidity = area / rect_area
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

        # 🚫 WINDOW FILTER (The Fix)
        # Rectangular items with high solidity and few vertices are ignored
        if len(approx) <= 6 and solidity > 0.7:
            continue 

        # ✅ GATE DETECTION
        # Gates (arcs) have complex shapes (high vertex count) and lower circularity
        if len(approx) > 10 and circularity < 0.6:
            aspect_ratio = w / float(h) if h != 0 else 0
            if 0.3 < aspect_ratio < 3:
                gate_list.append({
                    "x": int(x), 
                    "y": int(y), 
                    "w": int(w), 
                    "h": int(h),
                    "label": "gate"
                })

    return gate_list

get_Cordinates()