import cv2
import numpy as np
import os

# 1. Load File

def get_Cordinates():
    file_path = os.path.join(os.getcwd(), 'test', 'F2.png')
    img = cv2.imread(file_path)

    if img is None:
        print(f"Error: Could not find image at {file_path}")
        exit()

    # 2. Pre-processing Pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get white walls on black background
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove Noise & Text (Opening = Erosion then Dilation)
    kernel = np.ones((3,3), np.uint8)
    clean_walls = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Final touch: Median Blur to kill those "white dots" you mentioned
    clean_walls = cv2.medianBlur(clean_walls, 3)

    # 3. Detect Lines (CRITICAL: We use 'clean_walls' now, not 'thresh')
    line_img = img.copy()
    lines = cv2.HoughLinesP(clean_walls, 1, np.pi/180, threshold=40, 
                            minLineLength=60, maxLineGap=10)
    wall_list = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Filter for straight walls (Horizontal or Vertical)
            if abs(x1 - x2) < 10 or abs(y1 - y2) < 10:
                wall_data = {
                    "start": {"x": float(x1), "y": float(y1)},
                    "end": {"x": float(x2), "y": float(y2)},
                    # You can even round to 2 decimal places for cleanliness
                    "length": round(float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2)), 2)
                }
                wall_list.append(wall_data)
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    print("Detected Wall Coordinates:")
    print(wall_list)
    
    # 4. Create Tiled Display
    # Convert grayscale steps back to BGR so we can stack them with the color result
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    clean_bgr = cv2.cvtColor(clean_walls, cv2.COLOR_GRAY2BGR)

    # Stack images: Top row (Original & Threshold), Bottom row (Cleaned & Final)
    top_row = np.hstack((img, thresh_bgr))
    bottom_row = np.hstack((clean_bgr, line_img))
    combined_view = np.vstack((top_row, bottom_row))

    # Resize for screen fit if necessary
    cv2.imshow('Processing Steps (Clockwise: Orig, Thresh, Result, Clean)', 
               cv2.resize(combined_view, (0,0), fx=0.7, fy=0.7))

    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return wall_list