import cv2
import numpy as np
import json
import math

# ─────────────────────────────────────────────
# DETECTION HELPERS  (unchanged from original)
# ─────────────────────────────────────────────

def detect_walls(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("debug_walls", mask)  # Debug: save wall mask
    return mask


def detect_gates(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((100, 100), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    walls_mask = detect_walls(image_path)
    walls_mask = cv2.dilate(walls_mask, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.subtract(mask, walls_mask)
    for k_size in [5, 15]:
        h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, 1))
        v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_size))
        lines = cv2.add(
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, h_k),
            cv2.morphologyEx(mask, cv2.MORPH_OPEN, v_k),
        )
        mask = cv2.subtract(mask, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        extent = float(area) / (w * h)
        if 0.2 < (float(w) / h) < 5.0 and solidity < 0.6 and extent < 0.5:
            cv2.drawContours(clean_mask, [c], -1, 255, -1)
    return cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def detect_windows_by_shape(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    thick_mask = detect_walls(image_path)
    thick_mask_fat = cv2.dilate(thick_mask, np.ones((3, 3), np.uint8), iterations=1)
    thin_v = cv2.subtract(vertical_lines, thick_mask_fat)
    thin_h = cv2.subtract(horizontal_lines, thick_mask_fat)
    thin_lines = cv2.bitwise_or(thin_v, thin_h)
    return cv2.morphologyEx(thin_lines, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


# ─────────────────────────────────────────────
# NEW: GEOMETRY UTILITIES
# ─────────────────────────────────────────────

def contour_to_line_segment(c):
    """
    Fit a contour to its principal axis and return (start, end) as pixel coords.
    Works well for thin, elongated wall/window contours.
    """
    pts = c.reshape(-1, 2).astype(np.float32)
    mean, eigvec = cv2.PCACompute(pts, mean=None)
    center = mean[0]
    axis = eigvec[0]

    # Project all points onto the principal axis
    projs = [(np.dot(p - center, axis), p) for p in pts]
    projs.sort(key=lambda x: x[0])

    start = projs[0][1]
    end = projs[-1][1]
    return (
        {"x": int(round(start[0])), "y": int(round(start[1]))},
        {"x": int(round(end[0])),   "y": int(round(end[1]))},
    )


def bounding_rect_to_line(c):
    """
    Use the bounding box to derive start/end for roughly axis-aligned contours.
    """
    x, y, w, h = cv2.boundingRect(c)
    if w >= h:      # horizontal
        return {"x": x, "y": y + h // 2}, {"x": x + w, "y": y + h // 2}
    else:           # vertical
        return {"x": x + w // 2, "y": y}, {"x": x + w // 2, "y": y + h}


def point_on_wall(pt, start, end):
    """
    Project a point onto the wall line and return how far along the wall it sits
    (offset from start) and the wall's total length.
    """
    sx, sy = start["x"], start["y"]
    ex, ey = end["x"],   end["y"]
    px, py = pt["x"],    pt["y"]

    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length == 0:
        return 0, 0

    t = ((px - sx) * dx + (py - sy) * dy) / (length * length)
    offset = t * length
    return int(round(offset)), int(round(length))


def opening_width(c):
    """Estimate the width of an opening (gate/window) from its contour."""
    x, y, w, h = cv2.boundingRect(c)
    return max(w, h)


def find_nearest_wall(opening_center, wall_segments, threshold=30):
    """
    Return the index of the wall segment closest to `opening_center`.
    Returns None if the nearest wall is farther than `threshold` pixels.
    """
    best_idx, best_dist = None, float("inf")
    cx, cy = opening_center["x"], opening_center["y"]
    for i, (start, end, _) in enumerate(wall_segments):
        sx, sy = start["x"], start["y"]
        ex, ey = end["x"],   end["y"]
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length == 0:
            dist = math.hypot(cx - sx, cy - sy)
        else:
            t = max(0, min(1, ((cx - sx) * dx + (cy - sy) * dy) / (length * length)))
            nearest_x = sx + t * dx
            nearest_y = sy + t * dy
            dist = math.hypot(cx - nearest_x, cy - nearest_y)
        if dist < best_dist:
            best_dist, best_idx = dist, i
    return best_idx if best_dist <= threshold else None


def contour_center(c):
    M = cv2.moments(c)
    if M["m00"] == 0:
        x, y, w, h = cv2.boundingRect(c)
        return {"x": x + w // 2, "y": y + h // 2}
    return {"x": int(M["m10"] / M["m00"]), "y": int(M["m01"] / M["m00"])}


def segment_length(start, end):
    return math.hypot(end["x"] - start["x"], end["y"] - start["y"])


# ─────────────────────────────────────────────
# NEW: ROOM DETECTION via flood-fill on white space
# ─────────────────────────────────────────────

def detect_rooms(wall_mask, min_area=2000):
    """
    Invert the wall mask and flood-fill to find enclosed room regions.
    Returns a list of (name, bounding_box, mask) tuples.
    """
    h, w = wall_mask.shape
    # White space = potential rooms
    free = cv2.bitwise_not(wall_mask)
    # Clean tiny noise
    kernel = np.ones((5, 5), np.uint8)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, kernel)

    visited = np.zeros((h, w), dtype=np.uint8)
    rooms = []

    seed_step = 10
    room_id = 0
    for y in range(0, h, seed_step):
        for x in range(0, w, seed_step):
            if free[y, x] == 255 and visited[y, x] == 0:
                # Use a fresh copy of 'free' as the flood canvas (no mask buffer)
                tmp = free.copy()
                area = cv2.floodFill(tmp, None, (x, y), 128)[0]

                # The filled region is wherever tmp became 128
                region_mask = ((tmp == 128).astype(np.uint8)) * 255

                # Mark as visited regardless of size to avoid re-processing
                visited[region_mask > 0] = 1

                if area < min_area:
                    continue

                # Derive bounding box from the filled region
                cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                rx, ry, rw, rh = cv2.boundingRect(max(cnts, key=cv2.contourArea))

                room_id += 1
                rooms.append({
                    "name": f"Room {room_id}",
                    "bbox": {"x": rx, "y": ry, "w": rw, "h": rh},
                    "mask": region_mask,
                })

    return rooms


# ─────────────────────────────────────────────
# NEW: MAIN — extract_coordinates
# ─────────────────────────────────────────────

def extract_coordinates(image_path="test/F2.png", output_json="floor_plan_data.json"):
    """
    Analyse a floor-plan image and write a JSON file with the structure:

    {
      "project_name": "...",
      "image_size": {"width": W, "height": H},
      "rooms": [
        {
          "name": "Room N",
          "walls": [
            {
              "id": "Room N_W0",
              "start": {"x": ..., "y": ...},
              "end":   {"x": ..., "y": ...},
              "length": ...,
              "openings": [
                {"type": "window"|"gate", "offset": ..., "width": ...}
              ]
            }
          ]
        }
      ]
    }
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    img_h, img_w = img.shape[:2]

    # 1. Get binary masks
    wall_mask   = detect_walls(image_path)
    gate_mask   = detect_gates(image_path)
    window_mask = detect_windows_by_shape(image_path)

    # 2. Extract wall line segments
    wall_cnts, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_walls = []   # list of (start, end, contour)
    for c in wall_cnts:
        if cv2.contourArea(c) < 20:
            continue
        start, end = contour_to_line_segment(c)
        # Skip degenerate (near-zero-length) segments
        if segment_length(start, end) < 5:
            continue
        raw_walls.append((start, end, c))

    # 3. Collect openings (gates + windows) and attach to nearest wall
    openings_by_wall = {i: [] for i in range(len(raw_walls))}

    for opening_type, mask in [("gate", gate_mask), ("window", window_mask)]:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < 8:
                continue
            center = contour_center(c)
            width  = opening_width(c)
            wall_idx = find_nearest_wall(center, raw_walls, threshold=40)
            if wall_idx is None:
                continue
            start, end, _ = raw_walls[wall_idx]
            offset, _ = point_on_wall(center, start, end)
            openings_by_wall[wall_idx].append({
                "type":   opening_type,
                "offset": max(0, offset),
                "width":  int(width),
            })

    # 4. Detect rooms and assign walls to rooms
    rooms_detected = detect_rooms(wall_mask, min_area=2000)

    def wall_belongs_to_room(start, end, room_mask):
        """True if the midpoint of the wall lies on the boundary of the room."""
        mx = (start["x"] + end["x"]) // 2
        my = (start["y"] + end["y"]) // 2
        # Check a small neighbourhood around the mid-point for room-mask pixels
        r = 15
        patch = room_mask[
            max(0, my - r): min(img_h, my + r),
            max(0, mx - r): min(img_w, mx + r),
        ]
        return patch.any()

    result_rooms = []
    assigned = set()

    for room in rooms_detected:
        room_walls = []
        mask = room["mask"]
        for i, (start, end, _) in enumerate(raw_walls):
            if wall_belongs_to_room(start, end, mask):
                wall_id = f"{room['name'].replace(' ', '_')}_W{len(room_walls)}"
                room_walls.append({
                    "id":       wall_id,
                    "start":    start,
                    "end":      end,
                    "length":   int(round(segment_length(start, end))),
                    "openings": openings_by_wall.get(i, []),
                })
                assigned.add(i)

        if room_walls:
            result_rooms.append({
                "name":  room["name"],
                "walls": room_walls,
            })

    # 5. Any walls not yet assigned → put in an "Exterior / Unknown" room
    unassigned = [i for i in range(len(raw_walls)) if i not in assigned]
    if unassigned:
        ext_walls = []
        for i in unassigned:
            start, end, _ = raw_walls[i]
            wall_id = f"Exterior_W{len(ext_walls)}"
            ext_walls.append({
                "id":       wall_id,
                "start":    start,
                "end":      end,
                "length":   int(round(segment_length(start, end))),
                "openings": openings_by_wall.get(i, []),
            })
        result_rooms.append({"name": "Exterior / Unknown", "walls": ext_walls})

    # 6. Build final output dict
    output = {
        "project_name": image_path,
        "image_size":   {"width": img_w, "height": img_h},
        "rooms":        result_rooms,
    }

    # 7. Write JSON
    # with open(output_json, "w") as f:
    #     json.dump(output, f, indent=2)

    print(f"✅  Saved coordinate data to: {output_json}")
    print(f"    Rooms found : {len(result_rooms)}")
    print(f"    Total walls : {sum(len(r['walls']) for r in result_rooms)}")
    print(f"    Total openings: {sum(len(w['openings']) for r in result_rooms for w in r['walls'])}")
    return output


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    data = extract_coordinates(
        image_path="test/F2.png",
        output_json="floor_plan_data.json",
    )

    # Pretty-print a preview of the first room to stdout
    if data["rooms"]:
        print("\n--- Preview: first room ---")
        print(json.dumps(data["rooms"][0], indent=2))
        