import cv2
import numpy as np
import json

def get_wall_json(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "File not found"}
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. THRESHOLD & CLEAN
    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 100: 
            clean_mask[labels == i] = 255

    # 2. SKELETONIZE (Find the center line)
    skeleton = cv2.ximgproc.thinning(clean_mask) if hasattr(cv2, 'ximgproc') else clean_mask

    # 3. DETECT LINES
    lines = cv2.HoughLinesP(skeleton, 1, np.pi/180, threshold=20, 
                            minLineLength=20, maxLineGap=15)
    
    if lines is None:
        return {"project_name": "Empty Plan", "walls": []}

    raw_lines = [l[0] for l in lines]
    master_walls = []

    # 4. SNAP & MERGE (Remove double lines and segments)
    while len(raw_lines) > 0:
        l1 = raw_lines.pop(0)
        x1, y1, x2, y2 = l1
        is_h = abs(y1 - y2) < abs(x1 - x2)
        
        # Snap to 90 degrees
        if is_h: y2 = y1
        else: x2 = x1
        
        merged = False
        for i in range(len(master_walls)):
            mx1, my1, mx2, my2 = master_walls[i]
            m_is_h = abs(my1 - my2) < abs(mx1 - mx2)
            
            if is_h == m_is_h:
                dist = abs(y1 - my1) if is_h else abs(x1 - mx1)
                # If on same track and overlapping/near
                if dist < 12: 
                    if is_h:
                        if max(x1, x2) >= min(mx1, mx2) - 25 and min(x1, x2) <= max(mx1, mx2) + 25:
                            master_walls[i] = [min(x1, x2, mx1, mx2), my1, max(x1, x2, mx1, mx2), my1]
                            merged = True; break
                    else:
                        if max(y1, y2) >= min(my1, my2) - 25 and min(y1, y2) <= max(my1, my2) + 25:
                            master_walls[i] = [mx1, min(y1, y2, my1, my2), mx1, max(y1, y2, my1, my2)]
                            merged = True; break
        if not merged:
            master_walls.append([x1, y1, x2, y2])

    # 5. CONVERT TO DICTIONARY FORMAT
    wall_data = []
    for idx, wall in enumerate(master_walls):
        x1, y1, x2, y2 = wall
        wall_type = "horizontal" if abs(y1 - y2) < abs(x1 - x2) else "vertical"
        
        wall_data.append({
            "id": f"wall_{idx + 1}",
            "type": wall_type,
            "start": {"x": int(x1), "y": int(y1)},
            "end": {"x": int(x2), "y": int(y2)},
            "length": int(np.sqrt((x1-x2)**2 + (y1-y2)**2))
        })

    # FINAL STRUCTURE
    floorplan_json = {
        "project_info": {
            "name": "Floor Plan Extraction",
            "image_size": {"width": w, "height": h}
        },
        "walls": wall_data
    }

    return floorplan_json

# --- RUN AND SAVE ---
output_dict = get_wall_json('test/F2.png')

# Convert dictionary to JSON string and print
json_string = json.dumps(output_dict, indent=4)
print(json_string)
