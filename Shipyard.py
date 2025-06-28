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

# Configure Tesseract to use the local "tesseract" folder
script_dir = os.path.dirname(os.path.abspath(__file__))
exe_name = "tesseract.exe" if os.name == "nt" else "tesseract"
pytesseract.pytesseract.tesseract_cmd = os.path.join(script_dir, "tesseract", exe_name)


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
def processImage(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    pixels = image.load()
    w, h = image.size
    squares = []
    for y in range(h):
        for x in range(w):
            for off in range(5, 11):
                if (0 <= x < w and 0 <= y - off < h and
                    isColorWithinRange(pixels[x, y - off], (100, 100, 100), (0, 0, 0)) and
                    0 <= y + off < h and
                    isColorWithinRange(pixels[x, y + off], (100, 100, 100), (0, 0, 0)) and
                    0 <= x - off < w and
                    isColorWithinRange(pixels[x - off, y], (100, 100, 100), (0, 0, 0)) and
                    0 <= x + off < w and
                    isColorWithinRange(pixels[x + off, y], (100, 100, 100), (0, 0, 0))):
                    if testPixelsBetweenCenterAndOffset(pixels, x, y, off, w, h):
                        squares.append((x - off, y - off, 2 * off))
                        break
    valid = []
    for (tx, ty, d) in squares:
        off = d // 2
        if checkWallColor(pixels, tx + off, ty + off, off, w, h):
            valid.append((tx, ty, d))
    valid = removeOverlappingSquares(valid)
    valid = filterSmallSquares(valid)
    return image, valid

def isColorWithinRange(color, upper, lower):
    return all(lower[i] <= color[i] <= upper[i] for i in range(3))

def testPixelsBetweenCenterAndOffset(pixels, cx, cy, off, w, h):
    for i in range(1, off):
        if (isPixelOutsideRange(pixels, cx, cy - i, w, h) or
            isPixelOutsideRange(pixels, cx, cy + i, w, h) or
            isPixelOutsideRange(pixels, cx - i, cy, w, h) or
            isPixelOutsideRange(pixels, cx + i, cy, w, h)):
            return True
    return False

def isPixelOutsideRange(pixels, x, y, w, h):
    if 0 <= x < w and 0 <= y < h:
        return not isColorWithinRange(pixels[x, y], (100, 100, 100), (0, 0, 0))
    return False

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
    thr = 0.8 * avg
    return [sq for sq in squares if sq[2] >= thr]


# ===================== ShipyardText =====================
def displayImagesInShapeDifference(inputImage, shape1CoordsList, boxList):
    grouped_output = []
    highlighted_images = []
    if isinstance(inputImage, Image.Image):
        inputImage = np.array(inputImage)
    if len(inputImage.shape) == 3 and inputImage.shape[2] == 3:
        inputImage = cv2.cvtColor(inputImage, cv2.COLOR_RGB2BGR)
    if isinstance(inputImage, str):
        inputImage = cv2.imread(inputImage)
    for shape1Coords in shape1CoordsList:
        mask = np.zeros(inputImage.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(shape1Coords)], 255)
        masked = cv2.bitwise_and(inputImage, inputImage, mask=mask)
        for (x, y, s) in boxList:
            tl = (max(x-3,0), max(y-3,0))
            br = (min(x+s+3, inputImage.shape[1]), min(y+s+3, inputImage.shape[0]))
            cv2.rectangle(masked, tl, br, (255,255,255), -1)
        xs = [c[0] for c in shape1Coords]; ys = [c[1] for c in shape1Coords]
        xmin, xmax = max(min(xs),0), min(max(xs), inputImage.shape[1])
        ymin, ymax = max(min(ys),0), min(max(ys), inputImage.shape[0])
        if xmin < xmax and ymin < ymax:
            crop = masked[ymin:ymax, xmin:xmax]
            if crop.size>0:
                highlighted_images.append(crop)
            else:
                print("Warning: empty crop")
        else:
            print("Warning: invalid bbox")
    return grouped_output


# ===================== OCR Matching =====================
def matchTextToGroups(pil_image, group_polygons, threshold=50):
    """
    For each group polygon, find the closest merged OCR text (if within threshold)
    and draw a single line+label. Returns dict { group_index: text_or_None }.
    """
    # 1) run tesseract
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

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
        # Returns True if a and b overlap (with optional padding)
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
        # Merge group into one box
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
            'cy': (min_y + max_y) / 2
        })
        used.update(group)

    # 4) For each group, pick the closest merged text (within threshold), each text only once
    candidates = []
    for idx, m in enumerate(merged):
        for gi, poly in enumerate(group_polygons):
            minX, minY = poly[0]
            maxX, maxY = poly[2]
            cx, cy = m['cx'], m['cy']
            dx = max(minX - cx, 0, cx - maxX)
            dy = max(minY - cy, 0, cy - maxY)
            dist = math.hypot(dx, dy)
            if dist <= threshold:
                candidates.append({
                    'text': m['text'],
                    'cx': cx,
                    'cy': cy,
                    'group': gi,
                    'dist': dist,
                    'merged_idx': idx
                })

    best_for_group = {}
    used_merged_indices = set()
    sorted_candidates = sorted(candidates, key=lambda c: c['dist'])
    for c in sorted_candidates:
        gi = c['group']
        idx = c['merged_idx']
        if idx not in used_merged_indices and gi not in best_for_group:
            best_for_group[gi] = c
            used_merged_indices.add(idx)

    # 5) draw and build mapping
    mapping = {gi: None for gi in range(len(group_polygons))}
    draw = ImageDraw.Draw(pil_image)
    for gi, c in best_for_group.items():
        mapping[gi] = c['text']
        poly = group_polygons[gi]
        # Highlight the group polygon in pink
        draw.polygon(poly, outline="pink", width=3)
        # draw a line + label
        gx = sum(p[0] for p in poly)/4
        gy = sum(p[1] for p in poly)/4
        draw.line([(c['cx'], c['cy']), (gx, gy)], fill="blue", width=1)
        draw.text((c['cx'], c['cy']), c['text'], fill="green")

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
    img2, groups, polys, _ = connectNearbyBoxes(img1, valids)
    mapping = matchTextToGroups(img2, polys, threshold=50)
    print("Text → Group assignments:", mapping)
    draw = ImageDraw.Draw(img2)
    for grp in groups:
        for (x, y, s) in grp:
            draw.rectangle([x, y, x+s, y+s], outline="red", width=2)
    return img2, groups, polys, mapping

current_box_groups = None
current_text_mapping = None
labels_file_path = None
image_label = None

def open_and_process_image():
    global current_box_groups, current_text_mapping
    img = openAndConvertToColor()
    if not img: return
    img2, groups, polys, mapping = Processing(img)
    current_box_groups = (img2, groups, polys)
    current_text_mapping = mapping
    photo = ImageTk.PhotoImage(img2)
    image_label.config(image=photo); image_label.image = photo

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

def select_labels_file():
    global labels_file_path
    path = filedialog.askopenfilename(
        title="Select Labels File",
        filetypes=[("Text Files", "*.txt")]
    )
    if path:
        labels_file_path = path
        messagebox.showinfo("Labels File", f"Selected:\n{path}")
    else:
        messagebox.showerror("Error", "No labels file selected.")

def label_selection_dialog(group_index, available_labels):
    dlg = tk.Toplevel(); dlg.title(f"Assign Label for Group {group_index+1}")
    tk.Label(dlg, text=f"Select label for Group {group_index+1}:").pack(pady=10)
    var = tk.StringVar(dlg); var.set(available_labels[0])
    tk.OptionMenu(dlg, var, *available_labels).pack(pady=5)
    def on_ok():
        dlg.result = var.get(); dlg.destroy()
    tk.Button(dlg, text="OK", command=on_ok).pack(pady=10)
    dlg.grab_set(); dlg.wait_window()
    return getattr(dlg, "result", None)

def assign_labels():
    global current_box_groups, labels_file_path
    if not current_box_groups:
        messagebox.showinfo("Assign Labels", "No box groups processed.")
        return
    if not labels_file_path:
        messagebox.showerror("Error", "Select labels file first.")
        return
    try:
        with open(labels_file_path) as f:
            labels = [l.strip() for l in f if l.strip()]
        if not labels:
            messagebox.showerror("Error", "No labels in file.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read labels file: {e}")
        return

    img, _, polys = current_box_groups
    draw = ImageDraw.Draw(img)
    for poly in polys:
        draw.polygon(poly, outline="yellow", width=3)
    group_labels = {}
    for i, poly in enumerate(polys):
        lab = label_selection_dialog(i, labels) or ""
        group_labels[i] = lab
        draw.text((poly[0][0], max(0, poly[0][1] - 15)), lab, fill="blue")
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo); image_label.image = photo

    detail = "Assigned Labels:\n\n" + "\n".join(f"Group {i+1}: {lab}" for i,lab in group_labels.items())
    messagebox.showinfo("Assigned Labels", detail)

def create_gui():
    root = tk.Tk(); root.title("Shipyard OCR Matcher")
    tk.Button(root, text="Load & Process Image", command=open_and_process_image).pack(pady=5)
    tk.Button(root, text="Display Box Groups",   command=show_box_groups).pack(pady=5)
    tk.Button(root, text="Show Text→Group Map",  command=show_text_mapping).pack(pady=5)
    tk.Button(root, text="Select Labels File",   command=select_labels_file).pack(pady=5)
    tk.Button(root, text="Assign Labels",        command=assign_labels).pack(pady=5)
    global image_label
    image_label = tk.Label(root); image_label.pack()
    root.mainloop()

if __name__ == "__main__":
    create_gui()
