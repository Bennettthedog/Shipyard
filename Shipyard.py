"""
ShipyardCombined.py

Combined OCR-enabled Shipyard box finder, grouper, and GUI.
"""

import os
import math
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pytesseract
import copy
import difflib
import json

# Configure Tesseract to use the local "tesseract" folder
script_dir = os.path.dirname(os.path.abspath(__file__))
exe_name = "tesseract.exe" if os.name == "nt" else "tesseract"
pytesseract.pytesseract.tesseract_cmd = os.path.join(script_dir, "tesseract", exe_name)

# ——————————————————————————————
# Load “#GroupName” headings from your Type list.txt
def load_type_groups():
    """Returns dict[group_name] → list of type-names under that heading."""
    type_file = os.path.join(script_dir, "Type list.txt")
    groups = {}
    current = None
    if not os.path.exists(type_file):
        return groups
    with open(type_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                current = line.lstrip("#").strip()
                groups[current] = []
            elif current:
                groups[current].append(line)
    return groups

def get_all_types():
    """Flatten all types from all groups into a single list."""
    type_groups = load_type_groups()
    return [t for group in type_groups.values() for t in group]

# ——————————————————————————————
# Remove this old export_boxes_to_json function (lines 35–88)
# def export_boxes_to_json(mapping, box_groups, subsystem="ssd"):
#     ... (delete this function) ...

# ===================== BoxGrouper =====================
def connectNearbyBoxes(image, boxes):
    draw = ImageDraw.Draw(image)
    internal = []

    def isWithin5Pixels(b1, b2):
        x1, y1, s1 = b1
        x2, y2, s2 = b2
        left1, right1 = x1, x1 + s1
        top1, bottom1 = y1, y1 + s1
        left2, right2 = x2, x2 + s2
        top2, bottom2 = y2, y2 + s2
        if abs(right1 - left2) <= 5 or abs(left1 - right2) <= 5:
            if not (bottom1 < top2 or top1 > bottom2):
                return True
        if abs(bottom1 - top2) <= 5 or abs(top1 - bottom2) <= 5:
            if not (right1 < left2 or left1 > right2):
                return True
        return False

    def getCorners(b):
        x, y, s = b
        return [(x, y), (x + s, y), (x + s, y + s), (x, y + s)]

    def dfs(box, group, visited):
        visited.add(box)
        group.append(box)
        for other in boxes:
            if other not in visited and isWithin5Pixels(box, other):
                dfs(other, group, visited)

    groups = []
    visited = set()
    for b in boxes:
        if b not in visited:
            grp = []
            dfs(b, grp, visited)

            # --- New logic: search up to one box width away for unknown boxes ---
            added = True
            while added:
                added = False
                max_box_size = max(box[2] for box in grp) if grp else 0
                to_add = []
                for other in boxes:
                    if other in visited:
                        continue
                    ox, oy, os = other
                    for bx, by, bs in grp:
                        dist = ((ox + os/2) - (bx + bs/2))**2 + ((oy + os/2) - (by + bs/2))**2
                        if dist <= (max_box_size ** 2):
                            to_add.append(other)
                            break
                for box_to_add in to_add:
                    visited.add(box_to_add)
                    grp.append(box_to_add)
                    added = True

            groups.append(grp)

    for grp in groups:
        pts = [pt for b in grp for pt in getCorners(b)]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        minX, maxX = min(xs), max(xs)
        minY, maxY = min(ys), max(ys)
        poly = [(minX, minY), (maxX, minY), (maxX, maxY), (minX, maxY)]
        draw.polygon(poly, outline="red", width=2)
        internal.append(poly)

    return image, groups, internal, []


# ===================== ShipyardBoxFinder =====================
def findBlackOutlineBoxes(pil_image, square_tolerance=0.1, outlier_multiplier=1.5):
    """
    Finds black-outline boxes that are (approximately) square,
    then removes size outliers based on IQR of their widths.
    Accepts a PIL image and returns a list of (x, y, s) tuples.
    """
    # Convert PIL image to OpenCV format
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert: black becomes white and vice versa
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    # First pass: get all quads that look like squares
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h
        contour_area = cv2.contourArea(cnt)
        fill_ratio = contour_area / rect_area if rect_area > 0 else 0
        ratio = w / float(h)

        # keep boxes that are ≥50% filled, roughly square
        if fill_ratio >= 0.5 and abs(ratio - 1.0) <= square_tolerance:
            s = max(w, h)
            candidates.append((x, y, s))

    # If no candidates, skip outlier step
    if not candidates:
        return []

    # Outlier removal based on width IQR
    widths = np.array([s for (_, _, s) in candidates])
    q1, q3 = np.percentile(widths, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - outlier_multiplier * iqr
    upper_bound = q3 + outlier_multiplier * iqr

    boxes = [
        box for box in candidates
        if lower_bound <= box[2] <= upper_bound
    ]

    return boxes

import re   # add this at the top with your other imports

import re       # for regex
import cv2      # for cv2.cvtColor
import numpy as np
from PIL import ImageDraw
import pytesseract

def processImage(image):
    """
    Detect black‑outline boxes, OCR each one (cropping out the borders),
    print a progress update for each box, fill any box containing text with
    solid black, and return:
      img2, groups, polys, mapping, box_texts
    """
    # 1. Find individual boxes
    boxes = findBlackOutlineBoxes(image, square_tolerance=0.1, outlier_multiplier=1.5)
    total = len(boxes)

    # 2. OCR each box for any text (cropping in by a tiny border)
    box_texts = {}
    for idx, (x, y, s) in enumerate(boxes, start=1):
        print(f"Searching box {idx}/{total} at (x={x}, y={y}, size={s})…")

        # inset crop by a few pixels so we don't include the rectangle border
        border = max(1, int(s * 0.03))  # ~3% of box size, at least 1px
        left   = x + border
        top    = y + border
        right  = x + s - border
        bottom = y + s - border

        # ensure valid crop
        if right <= left or bottom <= top:
            crop = image.crop((x, y, x + s, y + s))
        else:
            crop = image.crop((left, top, right, bottom))

        # convert to grayscale & OCR
        gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
        config = "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        raw = pytesseract.image_to_string(gray, config=config)

        # pick out the first alphanumeric character, if any
        m = re.search(r"[A-Za-z0-9]", raw)
        if m:
            box_texts[idx-1] = m.group(0)

    # 3. Report which boxes had text
    for idx, char in box_texts.items():
        x, y, s = boxes[idx]
        print(f"→ Box {idx}: Position=({x}, {y}), Size={s} contains text '{char}'")

    # 4. Group boxes and get polygons
    img2, groups, polys, _ = connectNearbyBoxes(image, boxes)

    # 5. Match group → type/text
    mapping = matchTextToGroups(img2, polys, groups, threshold=50)

    # 6. Fill any OCR’d box solid black
    draw = ImageDraw.Draw(img2)
    for idx in box_texts:
        x, y, s = boxes[idx]
        draw.rectangle([x, y, x + s, y + s], fill="black")

    return img2, groups, polys, mapping, box_texts

def isColorWithinRange(color, upper, lower):
    return all(lower[i] <= color[i] <= upper[i] for i in range(3))

def checkWallColor(pixels, x, y, off, w, h):
    wallRange = ((100,100,100),(0,0,0))
    segs = {'top':[], 'bottom':[], 'left':[], 'right':[]}
    for i in range(x - off, x + off + 1):
        if 0 <= i < w:
            if 0 <= y - off < h: segs['top'].append(pixels[i, y - off])
            if 0 <= y + off < h: segs['bottom'].append(pixels[i, y + off])
    for i in range(y - off, y + off + 1):
        if 0 <= i < h:
            if 0 <= x - off < w: segs['left'].append(pixels[x - off, i])
            if 0 <= x + off < w: segs['right'].append(pixels[x + off, i])
    for seg in segs.values():
        count = sum(1 for px in seg if isColorWithinRange(px, wallRange[0], wallRange[1]))
        if len(seg) == 0 or count < 0.9 * len(seg):
            return False
    return True

def removeOverlappingSquares(squares):
    out = []
    used = set()
    for i in range(len(squares)):
        if i in used: continue
        keep = True
        for j in range(i+1, len(squares)):
            if calculateOverlap(squares[i], squares[j]) >= 0.9:
                keep = False
                used.add(j)
        if keep: out.append(squares[i])
    return out

def calculateOverlap(a, b):
    x1,y1,d1 = a; x2,y2,d2 = b
    l = max(x1, x2); r = min(x1+d1, x2+d2)
    t = max(y1, y2); bm = min(y1+d1, y2+d2)
    if l < r and t < bm:
        return ((r-l)*(bm-t)) / min(d1*d1, d2*d2)
    return 0

def filterSmallSquares(squares):
    if not squares: return squares
    avg = sum(d for _,_,d in squares) / len(squares)
    thr = 0.5 * avg
    return [sq for sq in squares if sq[2] >= thr]

# ===================== OCR Matching =====================
def load_type_list():
    """Load type list from 'Type list.txt' in the same folder, ignoring lines starting with '#'."""
    type_file = os.path.join(script_dir, "Type list.txt")
    if not os.path.exists(type_file):
        return []
    with open(type_file, encoding="utf-8") as f:
        types = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    return types

def best_type_match(text, type_list, threshold=0.7):
    """Return best match from type_list if similarity >= threshold, else None."""
    best = None
    best_score = 0
    for t in type_list:
        score = difflib.SequenceMatcher(None, text.lower(), t.lower()).ratio()
        if score > best_score:
            best_score = score
            best = t
    if best_score >= threshold:
        return best
    return None

def matchTextToGroups(pil_image, group_polygons, box_groups, threshold=50):
    """
    For each group polygon, starting from the most bottom-right, find the closest merged OCR text
    (within 1.5x box size). Prioritize ungrouped text directly above, then directly below.
    Returns dict { group_index: text_or_None }.
    """
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(
        gray, 
        output_type=pytesseract.Output.DICT, 
        config="--psm 11"
    )

    # 2) collect all OCR text boxes outside of any group
    raw_boxes = []
    for i, txt in enumerate(data['text']):
        txt = txt.strip()
        if not txt:
            continue
        x, y = data['left'][i], data['top'][i]
        w, h = data['width'][i], data['height'][i]
        cx, cy = x + w/2, y + h/2

        # skip if center inside any group rect
        inside = False
        for poly in group_polygons:
            minX, minY = poly[0]
            maxX, maxY = poly[2]
            if minX <= cx <= maxX and minY <= cy <= maxY:
                inside = True
                break
        if inside:
            continue
        raw_boxes.append({'text': txt, 'x': x, 'y': y, 'w': w, 'h': h})

    # 3) Merge overlapping text boxes
    def boxes_overlap(a, b, pad=2):
        ax1, ay1, ax2, ay2 = a['x']-pad, a['y']-pad, a['x']+a['w']+pad, a['y']+a['h']+pad
        bx1, by1, bx2, by2 = b['x']-pad, b['y']-pad, b['x']+b['w']+pad, b['y']+b['h']+pad
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    merged = []
    used = set()
    for i, box in enumerate(raw_boxes):
        if i in used:
            continue
        group = [i]
        for j in range(i+1, len(raw_boxes)):
            if j in used:
                continue
            if boxes_overlap(box, raw_boxes[j]):
                group.append(j)
                used.add(j)
        xs = [raw_boxes[k]['x'] for k in group]
        ys = [raw_boxes[k]['y'] for k in group]
        ws = [raw_boxes[k]['x'] + raw_boxes[k]['w'] for k in group]
        hs = [raw_boxes[k]['y'] + raw_boxes[k]['h'] for k in group]
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(ws), max(hs)
        merged_text = " ".join(raw_boxes[k]['text'] for k in group)
        merged.append({
            'text': merged_text,
            'x': min_x,
            'y': min_y,
            'w': max_x - min_x,
            'h': max_y - min_y,
            'cx': (min_x + max_x) / 2,
            'cy': (min_y + max_y) / 2,
            'bottom': max_y,
            'top': min_y
        })
        used.update(group)

    # --- Sort groups by bottom then right (most bottom-right first) ---
    group_indices = list(range(len(group_polygons)))
    group_bboxes = []
    for poly in group_polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        minX, maxX = min(xs), max(xs)
        minY, maxY = min(ys), max(ys)
        group_bboxes.append((minX, minY, maxX, maxY))
    group_indices.sort(key=lambda i: (group_bboxes[i][3], group_bboxes[i][2]), reverse=True)

    mapping = {gi: None for gi in range(len(group_polygons))}
    assigned_texts = set()
    draw = ImageDraw.Draw(pil_image)

    # Use all types from all groups for matching
    type_list = get_all_types()

    for gi in group_indices:
        poly = group_polygons[gi]
        minX, minY, maxX, maxY = group_bboxes[gi]
        group_height = maxY - minY

        # Get max box size in this group
        if box_groups and gi < len(box_groups):
            max_box_size = max(b[2] for b in box_groups[gi]) if box_groups[gi] else 0
        else:
            max_box_size = 0
        max_distance = 1.5 * max_box_size if max_box_size > 0 else threshold

        # 1. Find ungrouped text directly above (within horizontal bounds, above group)
        above_candidates = []
        for idx, m in enumerate(merged):
            if idx in assigned_texts:
                continue
            if m['bottom'] <= minY and minX <= m['cx'] <= maxX and m['h'] <= group_height:
                dy = minY - m['bottom']
                dx = abs(m['cx'] - (minX + maxX) / 2)
                dist = math.hypot(dx, dy)
                if dist <= max_distance:
                    above_candidates.append((dist, idx, m))
        above_candidates.sort()
        if above_candidates:
            _, idx, m = above_candidates[0]
            matched = best_type_match(m['text'], type_list, threshold=0.7)
            mapping[gi] = matched
            assigned_texts.add(idx)
            draw.polygon(poly, outline="pink", width=3)
            gx = sum(p[0] for p in poly)/4
            gy = sum(p[1] for p in poly)/4
            draw.line([(m['cx'], m['cy']), (gx, gy)], fill="blue", width=1)
            if matched:
                draw.text((m['cx'], m['cy']), matched, fill="green")
            continue

        # 2. If none, find ungrouped text directly below (within horizontal bounds, below group)
        below_candidates = []
        for idx, m in enumerate(merged):
            if idx in assigned_texts:
                continue
            if m['top'] >= maxY and minX <= m['cx'] <= maxX and m['h'] <= group_height:
                dy = m['top'] - maxY
                dx = abs(m['cx'] - (minX + maxX) / 2)
                dist = math.hypot(dx, dy)
                if dist <= max_distance:
                    below_candidates.append((dist, idx, m))
        below_candidates.sort()
        if below_candidates:
            _, idx, m = below_candidates[0]
            matched = best_type_match(m['text'], type_list, threshold=0.7)
            mapping[gi] = matched
            assigned_texts.add(idx)
            draw.polygon(poly, outline="orange", width=3)
            gx = sum(p[0] for p in poly)/4
            gy = sum(p[1] for p in poly)/4
            draw.line([(m['cx'], m['cy']), (gx, gy)], fill="purple", width=1)
            if matched:
                draw.text((m['cx'], m['cy']), matched, fill="green")
            continue

        # 3. Fallback: closest text (original logic)
        fallback_candidates = []
        for idx, m in enumerate(merged):
            if idx in assigned_texts:
                continue
            if m['h'] > group_height:
                continue
            cx, cy = m['cx'], m['cy']
            dx = max(minX - cx, 0, cx - maxX)
            dy = max(minY - cy, 0, cy - maxY)
            dist = math.hypot(dx, dy)
            if dist <= max_distance:
                fallback_candidates.append((dist, idx, m))
        fallback_candidates.sort()
        if fallback_candidates:
            _, idx, m = fallback_candidates[0]
            matched = best_type_match(m['text'], type_list, threshold=0.7)
            mapping[gi] = matched
            assigned_texts.add(idx)
            draw.polygon(poly, outline="gray", width=3)
            gx = sum(p[0] for p in poly)/4
            gy = sum(p[1] for p in poly)/4
            draw.line([(m['cx'], m['cy']), (gx, gy)], fill="gray", width=1)
            if matched:
                draw.text((m['cx'], m['cy']), matched, fill="green")

    return mapping


# ===================== Main & GUI =====================
def openAndConvertToColor():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
    )
    return Image.open(path) if path else None

def Processing(image):
    img1, valids = processImage(image)
    img2, groups, polys, mapping = Processing(img)
    current_box_groups = (img2, groups, polys)
    current_text_mapping = mapping
    photo = ImageTk.PhotoImage(img2)
    image_label.config(image=photo); image_label.image = photo

def Processing(image):
    # Use the new black-outline box finder
    valids = findBlackOutlineBoxes(image, square_tolerance=0.1, outlier_multiplier=1.5)
    img2, groups, polys, _ = connectNearbyBoxes(image, valids)
    mapping = matchTextToGroups(img2, polys, groups, threshold=50)  # <-- pass groups as box_groups
    print("Text → Group assignments:", mapping)
    draw = ImageDraw.Draw(img2)
    for grp in groups:
        for (x, y, s) in grp:
            draw.rectangle([x, y, x+s, y+s], outline="red", width=2)
    # ─── NEW: export JSON ───────────────────────
    # ───────────────────────────────────────────────
    return img2, groups, polys, mapping

current_box_groups = None
current_text_mapping = None
image_label = None

def open_and_process_image():
    global current_box_groups, current_text_mapping, image_label
    img = openAndConvertToColor()
    if not img:
        return
    # Use processImage to process the image and get labeled GUI image
    img2, groups, polys, mapping, _ = processImage(img)
    # Only highlight the groups in red
    draw = ImageDraw.Draw(img2)
    for poly in polys:
        draw.polygon(poly, outline="red", width=4)
    current_box_groups = (img2, groups, polys)
    current_text_mapping = mapping
    photo = ImageTk.PhotoImage(img2)
    image_label.config(image=photo)
    image_label.image

def show_box_groups():
    global current_box_groups
    if not current_box_groups:
        messagebox.showinfo("Box Groups", "No image processed yet.")
        return
    img, groups, _ = current_box_groups
    draw = ImageDraw.Draw(img)
    details = "Detected Box Groups:\n\n"
    for i, grp in enumerate(groups, start=1):
        xs = [b[0] for b in grp] + [b[0]+b[2] for b in grp]
        ys = [b[1] for b in grp] + [b[1]+b[2] for b in grp]
        minX, maxX = min(xs), max(xs)
        minY, maxY = min(ys), max(ys)
        draw.rectangle([minX, minY, maxX, maxY], outline="green", width=2)
        details += f"Group {i}: {grp}\n"
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo); image_label.image = photo
    messagebox.showinfo("Box Groups", details)

def show_text_mapping():
    global current_text_mapping
    if current_text_mapping is None:
        messagebox.showinfo("Text → Groups", "Process an image first.")
        return
    msg = ""
    for gi, texts in current_text_mapping.items():
        msg += f"Group {gi+1}: {texts}\n"
    messagebox.showinfo("Text → Groups", msg)

def show_cropped_ocr_images():
    global current_box_groups
    if not current_box_groups:
        messagebox.showinfo("Cropped OCR", "No image processed yet.")
        return
    img, groups, polys = current_box_groups
    crops = ocr_group_crops(img, polys, expand_factor=1.3)
    for idx, entry in enumerate(crops):
        crop_img = entry['crop']
        ocr_results = entry['ocr']
        win = tk.Toplevel()
        win.title(f"Group {idx+1} Cropped OCR")
        photo = ImageTk.PhotoImage(crop_img)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack()
        # Show OCR results as text
        text = ""
        for o in ocr_results:
            text += f"Text: {o['text']}\nBBoxes: {o['all_bboxes']}\n\n"
        tk.Label(win, text=text, justify="left", anchor="w").pack()
        tk.Button(win, text="Next", command=win.destroy).pack(pady=10)
        win.grab_set()
        win.wait_window()

def export_boxes_to_json(mapping, box_groups,
                         subsystem=None,
                         base_json="THO_NCL.json",
                         out_path=None):
    import os, json

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, base_json)

    # 1) Load existing template or initialize
    if os.path.exists(base_path):
        with open(base_path, 'r', encoding='utf-8') as f:
            base = json.load(f)
    else:
        base = {'ssd': {}}

    # Determine subsystem name
    if subsystem is None:
        subsystem = os.path.splitext(os.path.basename(base_json))[0]

    # Load group definitions
    type_groups = load_type_groups()
    type_groups_lc = {
        g.lower(): [t.lower() for t in types]
        for g, types in type_groups.items()
    }

    new_ssd = {}

    # 2) Torp, Drone, Phaser → one entry per box (unchanged)
    for cat in ('torp', 'drone', 'phaser'):
        members = type_groups_lc.get(cat, [])
        entries = []
        for gi, t in mapping.items():
            if t and t.lower() in members:
                for x, y, s in box_groups[gi]:
                    cx = int(x + s/2)
                    cy = int(y + s/2)
                    entries.append({
                        'type': t,
                        'designation': '',
                        'pos': f"{cx},{cy}",
                        'arc': ''
                    })
        new_ssd[cat] = entries

    # 3) Other category → export each subtype separately, skip empty lists
    for subtype in type_groups_lc.get('other', []):
        positions = []
        for gi, t in mapping.items():
            if t and t.lower() == subtype:
                for x, y, s in box_groups[gi]:
                    cx = int(x + s/2)
                    cy = int(y + s/2)
                    positions.append({'pos': f"{cx},{cy}"})
        if positions:
            new_ssd[subtype] = positions

    # 4) Preserve sensor/scanner/damcon/excessdamage if present
    for special in ('sensor', 'scanner', 'damcon', 'excessdamage'):
        if special in base.get('ssd', {}):
            new_ssd[special] = base['ssd'][special]

    # 5) Shields → counts per subtype (moved to bottom)
    shield_counts = {}
    for gi, t in mapping.items():
        if t and t.lower() in type_groups_lc.get('shield', []):
            key = t.lower().replace(' ', '_')
            shield_counts[key] = shield_counts.get(key, 0) + len(box_groups[gi])
    if shield_counts:
        new_ssd['shields'] = shield_counts

    # 6) Write out
    base['ssd'] = new_ssd
    if out_path is None:
        out_path = os.path.join(script_dir, f"{subsystem}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=4)

    print(f"Exported → {out_path}")


def export_json_gui():
    global current_text_mapping, current_box_groups
    # 1) make sure we’ve processed at least one image
    if current_text_mapping is None or current_box_groups is None:
        messagebox.showinfo("Export", "Process an image first.")
        return

    try:
        # 2) actually call your existing export function
        export_boxes_to_json(
            current_text_mapping,
            current_box_groups[1]   # pass just the 'groups' list
        )
        messagebox.showinfo("Export", "Exported JSON successfully.")
    except Exception as e:
        # 3) catch anything (FileNotFound, JSON errors, etc.) and display it
        messagebox.showerror("Export Error", f"Failed to export JSON:\n{e}")

def create_gui():
    root = tk.Tk(); root.title("Shipyard OCR Matcher")
    tk.Button(root, text="Load & Process Image", command=open_and_process_image).pack(pady=5)
    tk.Button(root, text="Show Text→Group Map",  command=show_text_mapping).pack(pady=5)
    tk.Button(root, text="Show Cropped OCR Images", command=show_cropped_ocr_images).pack(pady=5)
    tk.Button(root,
              text="Export as JSON",
              command=export_json_gui
             ).pack(pady=5)

    global image_label
    image_label = tk.Label(root); image_label.pack()

    # Ensure the app fully exits when the window is closed
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()
    os._exit(0)  # Force exit to kill any background threads or windows

if __name__ == "__main__":
    create_gui()

def get_group_surrounding_images(pil_image, group_polygons, box_groups, box_length_factor=1.5):
    """
    For each group, returns a masked PIL image where:
      - The group itself is masked out (black).
      - Only the area within (box_length_factor * box size) from the edge of each box in the group is visible.
      - All other areas are masked out (black).
    Returns: list of PIL.Image objects, one per group.
    """
    images = []
    img_w, img_h = pil_image.size

    for group_idx, poly in enumerate(group_polygons):
        group_boxes = box_groups[group_idx]
        # Create mask: black everywhere
        mask = Image.new("L", (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)
        # For each box in the group, draw an expanded rectangle in white
        for (x, y, s) in group_boxes:
            pad = int(box_length_factor * s)
            exp_minX = max(0, x - pad)
            exp_maxX = min(img_w, x + s + pad)
            exp_minY = max(0, y - pad)
            exp_maxY = min(img_h, y + s + pad)
            draw.rectangle([exp_minX, exp_minY, exp_maxX, exp_maxY], fill=255)
        # Draw group polygon black (mask out group itself)
        draw.polygon(poly, fill=0)

        # Apply mask to image
        masked = pil_image.copy()
        masked_np = np.array(masked)
        mask_np = np.array(mask)
        masked_np[mask_np == 0] = 0
        masked_img = Image.fromarray(masked_np)

        images.append(masked_img)

    return images

def show_group_masks(pil_image, group_polygons, box_groups, box_length_factor=1.5):
    """
    Display the mask for each group one by one in a Tkinter window.
    """
    images = get_group_surrounding_images(pil_image, group_polygons, box_groups, box_length_factor)
    for idx, img in enumerate(images):
        win = tk.Toplevel()
        win.title(f"Group {idx+1} Mask")
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo)
        lbl.image = photo
        lbl.pack()
        tk.Label(win, text=f"Group {idx+1}").pack()
        tk.Button(win, text="Next", command=win.destroy).pack(pady=10)
        win.grab_set()
        win.wait_window()

def ocr_group_crops(pil_image, group_polygons, expand_factor=1.3):
    """
    For each group:
      - Crop to 1.3x group bounding box.
      - Mask all group polygons to white.
      - Run OCR, save text and true location (original image coordinates).
      - Merge overlapping texts, keep all possible bounding boxes.
    Returns: List of dicts, one per group:
      { 'crop': PIL.Image, 'ocr': [ { 'text', 'orig_bbox', 'all_bboxes' } ] }
    """
    results = []
    img_w, img_h = pil_image.size

    # Precompute all group bounding boxes
    group_bboxes = []
    for poly in group_polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        minX, maxX = min(xs), max(xs)
        minY, maxY = min(ys), max(ys)
        group_bboxes.append((minX, minY, maxX, maxY))

    for idx, (poly, bbox) in enumerate(zip(group_polygons, group_bboxes)):
        minX, minY, maxX, maxY = bbox
        cx, cy = (minX + maxX) / 2, (minY + maxY) / 2
        w, h = maxX - minX, maxY - minY
        pad_w, pad_h = (expand_factor - 1) * w / 2, (expand_factor - 1) * h / 2
        crop_minX = max(0, int(cx - w/2 - pad_w))
        crop_maxX = min(img_w, int(cx + w/2 + pad_w))
        crop_minY = max(0, int(cy - h/2 - pad_h))
        crop_maxY = min(img_h, int(cy + h/2 + pad_h))

        # Crop and mask
        crop_box = (crop_minX, crop_minY, crop_maxX, crop_maxY)
        crop_img = pil_image.crop(crop_box).convert("RGB")
        mask = Image.new("L", crop_img.size, 255)
        draw = ImageDraw.Draw(mask)
        # Draw all group polygons (shifted to crop coords) as white (mask out)
        for gpoly in group_polygons:
            shifted = [(x-crop_minX, y-crop_minY) for (x, y) in gpoly]
            draw.polygon(shifted, fill=255)
        # Apply mask: set masked areas to white
        crop_np = np.array(crop_img)
        mask_np = np.array(mask)
        crop_np[mask_np == 255] = 255
        crop_img = Image.fromarray(crop_np)

        # OCR
        ocr_data = pytesseract.image_to_data(
            crop_img, 
            output_type=pytesseract.Output.DICT, 
            config="--psm 11"
        )
        texts = []
        for i, txt in enumerate(ocr_data['text']):
            txt = txt.strip()
            if not txt:
                continue
            x, y = ocr_data['left'][i], ocr_data['top'][i]
            w, h = ocr_data['width'][i], ocr_data['height'][i]
            # Map to original image coords
            orig_bbox = (x + crop_minX, y + crop_minY, x + crop_minX + w, y + crop_minY + h)
            texts.append({'text': txt, 'orig_bbox': orig_bbox, 'bbox': (x, y, x+w, y+h)})

        # Merge overlapping texts
        merged = []
        used = set()
        def overlap(a, b, pad=2):
            ax1, ay1, ax2, ay2 = a['orig_bbox']
            bx1, by1, bx2, by2 = b['orig_bbox']
            return not (ax2+pad < bx1 or ax1-pad > bx2 or ay2+pad < by1 or ay1-pad > by2)

        for i, t in enumerate(texts):
            if i in used: continue
            group_idxs = [i]
            for j in range(i+1, len(texts)):
                if j in used: continue
                if overlap(t, texts[j]):
                    group_idxs.append(j)
                    used.add(j)
            all_bboxes = [texts[k]['orig_bbox'] for k in group_idxs]
            merged_text = " ".join(texts[k]['text'] for k in group_idxs)
            merged.append({
                'text': merged_text,
                'orig_bbox': all_bboxes[0],  # first bbox
                'all_bboxes': all_bboxes
            })
            used.update(group_idxs)

        results.append({
            'crop': crop_img,
            'ocr': merged
        })
    return results

# sheild results arent working
# sorting isnt working
