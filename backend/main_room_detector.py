import cv2
import numpy as np
import os
import json
from shapely.geometry import LineString, Polygon, Point


# ═══════════════════════════════════════════════════════════════════
#  TUNABLE CONSTANTS  — adjust these if your image scale changes
# ═══════════════════════════════════════════════════════════════════

HC_DP          = 1
HC_MIN_DIST    = 30
HC_PARAM1      = 50
HC_PARAM2      = 20
HC_MIN_RADIUS  = 30
HC_MAX_RADIUS  = 80

PIVOT_SNAP        = 25
RADIUS_TOLERANCE  = 0.40
WIN_SNAP          = 14


# ═══════════════════════════════════════════════════════════════════
#  1. WINDOW DETECTION
# ═══════════════════════════════════════════════════════════════════

def _detect_windows(gray: np.ndarray) -> list[dict]:
    """Returns list of {x, y, w, h, cx, cy}."""
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    windows = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (100 < area < 5000):
            continue
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        if not (aspect > 1.5 or aspect < 0.6):
            continue
        if not cv2.isContourConvex(approx):
            continue
        windows.append({"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2})

    return windows


# ═══════════════════════════════════════════════════════════════════
#  2. GATE / DOOR DETECTION — HoughCircles-based
# ═══════════════════════════════════════════════════════════════════

def _detect_gates(gray: np.ndarray, debug_img=None) -> list[dict]:
    """Returns list of {cx, cy, width}."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    raw = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=HC_DP, minDist=HC_MIN_DIST, param1=HC_PARAM1, param2=HC_PARAM2,
        minRadius=HC_MIN_RADIUS, maxRadius=HC_MAX_RADIUS,
    )
    gates = []
    if raw is None:
        return gates
    for cx, cy, r in np.round(raw[0]).astype(int):
        gates.append({"cx": float(cx), "cy": float(cy), "width": float(r)})
        if debug_img is not None:
            cv2.circle(debug_img, (cx, cy), r, (0, 140, 255), 2)
            cv2.circle(debug_img, (cx, cy), 4, (0,  60, 255), -1)
    return gates


# ═══════════════════════════════════════════════════════════════════
#  3. CLASSIFY OPENINGS PER WALL SEGMENT
# ═══════════════════════════════════════════════════════════════════

def _classify_openings(line: LineString, windows: list[dict], gates: list[dict]) -> list[dict]:
    openings = []

    for win in windows:
        pt = Point(win["cx"], win["cy"])
        if line.distance(pt) <= WIN_SNAP:
            proj = line.project(pt)
            openings.append({"type": "window", "offset": round(proj, 2), "width": round(max(win["w"], win["h"]), 2)})
            break

    if openings:
        return openings

    p0, p1 = Point(line.coords[0]), Point(line.coords[1])
    line_len = line.length

    for gate in gates:
        gpt = Point(gate["cx"], gate["cy"])
        if not (gpt.distance(p0) <= PIVOT_SNAP or gpt.distance(p1) <= PIVOT_SNAP):
            continue
        if abs(gate["width"] - line_len) / max(line_len, 1) > RADIUS_TOLERANCE:
            continue
        openings.append({"type": "gate", "offset": 0, "width": round(gate["width"], 2)})
        break

    return openings


# ═══════════════════════════════════════════════════════════════════
#  4. ADVANCED ROOM DETECTOR — returns {"rooms": [...]}
#  Import this instead of main.py for Shapely-based room analysis.
# ═══════════════════════════════════════════════════════════════════

def get_rooms(DEBUG: bool = False) -> dict:
    """
    Advanced room/wall/opening detector using Shapely geometry.
    Used by test.py and other standalone scripts.
    The /api/data route uses turtle_test + t.py instead.
    """
    file_path = os.path.join(os.getcwd(), 'test', 'F2.png')
    img = cv2.imread(file_path)
    if img is None:
        return {"rooms": []}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    text_mask = np.zeros_like(thresh)
    for i in range(1, nlabels):
        if stats[i,2] < 60 and stats[i,3] < 60 and stats[i,4] < 800:
            text_mask[labels == i] = 255

    clean_img  = cv2.inpaint(img, text_mask, 3, cv2.INPAINT_TELEA)
    clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)

    _, clean_thresh = cv2.threshold(clean_gray, 200, 255, cv2.THRESH_BINARY_INV)
    walls_mask = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    windows = _detect_windows(clean_gray)
    debug_line_img = img.copy()
    gates = _detect_gates(clean_gray, debug_img=debug_line_img if DEBUG else None)

    contours, _ = cv2.findContours(walls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = cv2.HoughLinesP(walls_mask, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=15)

    rooms_data = []
    if lines is not None:
        all_wall_lines = [LineString([(l[0][0], l[0][1]), (l[0][2], l[0][3])]) for l in lines]

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 2000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            room_poly = Polygon(cnt.reshape(-1, 2))
            room_obj = {
                "name":   f"Room {i + 1}",
                "center": {"x": float(x + w / 2), "y": float(y + h / 2)},
                "walls":  [],
            }
            for line in all_wall_lines:
                if not room_poly.buffer(5).intersects(line):
                    continue
                x1, y1 = line.coords[0]
                x2, y2 = line.coords[1]
                openings = _classify_openings(line, windows, gates)
                room_obj["walls"].append({
                    "id":       f"wall_{len(room_obj['walls'])}",
                    "start":    {"x": float(x1), "y": float(y1)},
                    "end":      {"x": float(x2), "y": float(y2)},
                    "openings": openings,
                })
            rooms_data.append(room_obj)

    return {"rooms": rooms_data}


if __name__ == "__main__":
    result = get_rooms(DEBUG=True)
    print(json.dumps(result, indent=2))
