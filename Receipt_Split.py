"""
PDF Receipt & Invoice Segmentation Pipeline
-------------------------------------------
Author: Kyle TC-Singh

DESCRIPTION:
This script processes scanned PDF documents (e.g., receipts, invoices) containing 
multiple elements on a single page. It detects individual elements, draws bounding 
boxes for debugging, and splits them into separate PDF pages for structured analysis.

FEATURES:
- Auto-detection of receipts/invoices on each page
- Adaptive kernel tuning for better box detection
- Vertical stacking logic to group elements correctly
- Smart post-processing to merge small fragments (e.g. headers)
- Optional debug output with visual overlays and logging

SETUP INSTRUCTIONS:
1. Install required Python packages:
   pip install opencv-python pillow pdf2image numpy

2. You must also install Poppler for `pdf2image`:
   - Windows: https://github.com/oschwartz10612/poppler-windows
   - macOS:   brew install poppler
   - Linux:   sudo apt install poppler-utils

3. Optional: Create a virtual environment (recommended)
   python -m venv myenv
   myenv\Scripts\activate (Windows) or source myenv/bin/activate (macOS/Linux)

CONFIGURATION:
- Place your input PDFs in the 'input_folder' path.
- Output PDFs and debug data will be saved in 'output_folder'.
- Toggle `save_debug_output = True` to enable debug PDF + JSON logging.

USAGE:
- Run this script in your Python environment.
- Ensure paths are correct and you have write permission for the output folder.

"""

import os, cv2, json, numpy as np
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path

# --------------------------- CONFIG ----------------------------------
input_folder  = r"C:\Users\kyle_\Desktop\Visual Alpha\Multiple Receipt Analysis\VSC\PDFs\Training Data"
output_folder = r"C:\Users\kyle_\Desktop\Visual Alpha\Multiple Receipt Analysis\VSC\OpenCV\New Era Output 2"
os.makedirs(output_folder, exist_ok=True)

show_box_stats = False  # Set to True for detailed box size and location logs
save_debug_output = False # Set to True to save debug PDFs with bounding boxes
detection_log_path = os.path.join(output_folder, "combined_detection_log.jsonl")
dpi = 300

primary_kernel_size       = 50
secondary_kernel_size     = 42
min_box_area              = 150000
small_box_area_threshold  = 250000
max_allowed_small_boxes   = 3
cleanup_kernel_size       = 32
cleanup_min_area          = 30000
# ---------------------------------------------------------------------

def log_detection_results_global(pdf_name, page_index, boxes, is_fullpage, is_final, log_file_path):
    page_log = {
        "timestamp": datetime.now().isoformat(),
        "pdf_name": pdf_name,
        "page_number": page_index + 1,
        "full_page_detected": is_fullpage,
        "final": is_final,
        "detected_boxes": [
            {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
             "width": int(x2 - x1), "height": int(y2 - y1),
             "area": int((x2 - x1) * (y2 - y1))} for (x1, y1, x2, y2) in boxes]
    }
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(page_log) + "\n")

def detect_boxes(image_cv, kernel_size, min_area_override=None):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes, small_boxes = [], 0
    threshold = min_box_area if min_area_override is None else min_area_override

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > threshold:
            boxes.append((x, y, x + w, y + h))
            if area < small_box_area_threshold:
                small_boxes += 1

    is_fullpage = len(boxes) >= max_allowed_small_boxes and small_boxes >= max_allowed_small_boxes
    return boxes, is_fullpage

def apply_dominant_box_override(boxes, is_final, page_width, page_height):
    if is_final:
        return boxes, False

    page_area = page_width * page_height
    if len(boxes) == 1:
        x1, y1, x2, y2 = boxes[0]
        area = (x2 - x1) * (y2 - y1)
        if area > 0.85 * page_area:
            return [], True

    large_boxes = [b for b in boxes if (b[2] - b[0]) > 0.7 * page_width and (b[3] - b[1]) > 0.3 * page_height]
    if len(large_boxes) == 2:
        return boxes, False

    for (x1, y1, x2, y2) in boxes:
        if (x2 - x1) > 0.6 * page_width and (y2 - y1) > 0.4 * page_height:
            return [], True

    return boxes, False

def group_vertically_stacked_boxes(boxes, page_height, vertical_gap_threshold=90, vertical_axis_threshold=80):
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: b[1])
    grouped, used = [], [False] * len(boxes)

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        group = [boxes[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            bx1, by1, bx2, by2 = boxes[j]

            horizontal_overlap = max(0, min(x2, bx2) - max(x1, bx1))
            min_width = min(x2 - x1, bx2 - bx1)
            alignment_score = horizontal_overlap / min_width if min_width > 0 else 0
            width_ratio = min(x2 - x1, bx2 - bx1) / max(x2 - x1, bx2 - bx1)
            same_axis = abs(((x1 + x2) // 2) - ((bx1 + bx2) // 2)) < vertical_axis_threshold

            if (abs(by1 - y2) < vertical_gap_threshold and (by1 - y2) > -20
                    and alignment_score > 0.8 and width_ratio > 0.60 and same_axis):
                x1, y1 = min(x1, bx1), min(y1, by1)
                x2, y2 = max(x2, bx2), max(y2, by2)
                group.append(boxes[j])
                used[j] = True
        grouped.append((x1, y1, x2, y2))
    return grouped

def smart_merge_small_receipt_fragments(boxes, page_width, page_height):
    if len(boxes) < 2:
        return boxes

    used, merged = [False]*len(boxes), []

    for i in range(len(boxes)):
        if used[i]:
            continue
        x1, y1, x2, y2 = boxes[i]
        area_main, cx_main = (x2 - x1) * (y2 - y1), (x1 + x2) // 2

        for j in range(len(boxes)):
            if i == j or used[j]:
                continue
            bx1, by1, bx2, by2 = boxes[j]
            area_small, cx_small = (bx2 - bx1) * (by2 - by1), (bx1 + bx2) // 2

            if (area_small >= 0.5 * area_main
                    or min(abs(y2 - by1), abs(by2 - y1)) > 180
                    or abs(cx_main - cx_small) > 80):
                continue

            x1, y1 = min(x1, bx1), min(y1, by1)
            x2, y2 = max(x2, bx2), max(y2, by2)
            used[j] = True

        used[i] = True
        merged.append((x1, y1, x2, y2))
    return merged
def merge_small_yellow_boxes(boxes, page_width, page_height, area_threshold=350_000):
    if len(boxes) < 2:
        return boxes

    merged = []
    used = [False] * len(boxes)

    def compute_center(b):
        return ((b[0] + b[2]) // 2, (b[1] + b[3]) // 2)

    for i, box in enumerate(boxes):
        if used[i]:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area >= area_threshold:
            merged.append(box)
            used[i] = True
            continue

        cx1, cy1 = compute_center(box)
        min_dist = float('inf')
        closest_j = None
        for j, other in enumerate(boxes):
            if i == j or used[j]:
                continue
            cx2, cy2 = compute_center(other)
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_j = j

        if closest_j is not None:
            ox1, oy1, ox2, oy2 = boxes[closest_j]
            merged_box = (
                min(x1, ox1), min(y1, oy1),
                max(x2, ox2), max(y2, oy2)
            )
            merged.append(merged_box)
            used[i] = True
            used[closest_j] = True
        else:
            merged.append(box)
            used[i] = True

    return merged

def merge_small_boxes_by_edge_distance_only_small(boxes, area_threshold=350_000):
    if len(boxes) < 2:
        return boxes

    used = [False] * len(boxes)
    merged = []

    def overlaps_1d(a1, a2, b1, b2):
        return min(a2, b2) > max(a1, b1)

    for i, box in enumerate(boxes):
        if used[i]:
            continue

        x1a, y1a, x2a, y2a = box
        area = (x2a - x1a) * (y2a - y1a)

        if area >= area_threshold:
            merged.append(box)
            used[i] = True
            continue

        best_j = None
        min_distance = float('inf')

        for j, other in enumerate(boxes):
            if i == j or used[j]:
                continue
            x1b, y1b, x2b, y2b = other

            if overlaps_1d(y1a, y2a, y1b, y2b):
                dist_left = abs(x1a - x2b)
                dist_right = abs(x2a - x1b)
                min_h_dist = min(dist_left, dist_right)
                if min_h_dist < min_distance:
                    min_distance = min_h_dist
                    best_j = j

            if overlaps_1d(x1a, x2a, x1b, x2b):
                dist_top = abs(y1a - y2b)
                dist_bottom = abs(y2a - y1b)
                min_v_dist = min(dist_top, dist_bottom)
                if min_v_dist < min_distance:
                    min_distance = min_v_dist
                    best_j = j

        if best_j is not None:
            x1b, y1b, x2b, y2b = boxes[best_j]
            new_box = (min(x1a, x1b), min(y1a, y1b), max(x2a, x2b), max(y2a, y2b))
            merged.append(new_box)
            used[i] = True
            used[best_j] = True
        else:
            merged.append(box)
            used[i] = True
            
    return merged

def merge_remaining_small_boxes_to_nearest_large(boxes, area_threshold=350000):
    if len(boxes) < 2:
        return boxes

    small = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) < area_threshold]
    large = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) >= area_threshold]
    if not small or not large:
        return boxes

    def box_center(b):
        return ((b[0]+b[2])//2, (b[1]+b[3])//2)

    merged = []
    used = set()
    for s in small:
        sx, sy = box_center(s)
        nearest = None
        min_dist = float("inf")
        for l in large:
            lx, ly = box_center(l)
            dist = ((sx - lx)**2 + (sy - ly)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = l
        if nearest:
            new_box = (
                min(s[0], nearest[0]),
                min(s[1], nearest[1]),
                max(s[2], nearest[2]),
                max(s[3], nearest[3])
            )
            used.add(tuple(nearest))
            used.add(tuple(s))
            merged.append(new_box)

    final_boxes = [b for b in boxes if tuple(b) not in used]
    final_boxes.extend(merged)
    return final_boxes


    return merged

def draw_bounding_boxes_dynamic(image_cv):
    boxes, is_fullpage = detect_boxes(image_cv, primary_kernel_size)
    if len(boxes) == 1:
        boxes_retry, is_fullpage_retry = detect_boxes(image_cv, secondary_kernel_size)
        if len(boxes_retry) > 1:
            boxes, is_fullpage = boxes_retry, is_fullpage_retry

    is_final = False
    page_height, page_width = image_cv.shape[:2]
    large_boxes = [b for b in boxes if (b[2] - b[0]) > 0.7 * page_width and (b[3] - b[1]) > 0.3 * page_height]
    if len(large_boxes) == 2:
        is_final = True

    if not is_final:
        boxes = group_vertically_stacked_boxes(boxes, page_height)

    boxes, override_fullpage = apply_dominant_box_override(boxes, is_final, page_width, page_height)
    if override_fullpage:
        is_fullpage, is_final = True, True

    if not is_final and not is_fullpage:
        cleanup_boxes, _ = detect_boxes(image_cv, cleanup_kernel_size, cleanup_min_area)

        def nested(frag, existing):
            fx1, fy1, fx2, fy2 = frag
            return any(fx1 >= ex1 and fy1 >= ey1 and fx2 <= ex2 and fy2 <= ey2 for (ex1, ey1, ex2, ey2) in existing)

        eligible = [b for b in cleanup_boxes if (b[2]-b[0])*(b[3]-b[1]) < 200_000 and not nested(b, boxes)]
        vert_receipt_like = [b for b in boxes if (b[2]-b[0]) < 0.5*page_width and (b[3]-b[1]) > 0.4*page_height]

        if ((len(eligible) >= 2 and len(vert_receipt_like) < 2)
                or (len(eligible) == 1 and len(vert_receipt_like) == 0)):
            boxes, is_fullpage, is_final = [], True, True

    if not is_final and not is_fullpage:
        boxes = smart_merge_small_receipt_fragments(boxes, page_width, page_height)
        boxes = merge_small_yellow_boxes(boxes, page_width, page_height)
        boxes = merge_small_boxes_by_edge_distance_only_small(boxes)

    out = image_cv.copy()
    if is_final and not is_fullpage:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 4)
    elif not is_fullpage:
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,255), 3)
            center_x = (x1 + x2)//2
            cv2.line(out, (center_x, y1), (center_x, y2), (255,0,255), 2)
            
    # Final safety merge: only run if leftover small boxes exist
    if not is_fullpage and not is_final:
        small_boxes = [b for b in boxes if (b[2]-b[0])*(b[3]-b[1]) < 350000]
        if small_boxes:
            boxes = merge_remaining_small_boxes_to_nearest_large(boxes)


    return out, boxes, is_fullpage, is_final

# ---------------------------- MAIN LOOP ------------------------------
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(input_folder, filename)
    debug_name = filename.replace(".pdf", "_debug.pdf")
    split_name = filename.replace(".pdf", "_split.pdf")
    debug_path = os.path.join(output_folder, debug_name)
    split_path = os.path.join(output_folder, split_name)

    n = 1
    while os.path.exists(debug_path):
        debug_path = os.path.join(output_folder, filename.replace(".pdf", f"_debug_{n}.pdf"))
        split_path = os.path.join(output_folder, filename.replace(".pdf", f"_split_{n}.pdf"))
        n += 1

    print(f"Processing: {filename}")
    orig_pages = convert_from_path(pdf_path, dpi=dpi)
    debug_pages, split_pages = [], []
    full_page_indices = []

    for idx, orig_pil in enumerate(orig_pages):
        cv_img = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
        boxed_img, boxes, is_fullpage, is_final = draw_bounding_boxes_dynamic(cv_img)

        if show_box_stats:
            print(f"Page {idx+1} in {filename} — Box Summary:")
            for i, (x1, y1, x2, y2) in enumerate(boxes):
                width  = x2 - x1
                height = y2 - y1
                cx     = (x1 + x2) // 2
                cy     = (y1 + y2) // 2
                area   = width * height
                print(f"   Box {i+1}: (x1={x1}, y1={y1}, x2={x2}, y2={y2}) → W={width}, H={height}, Area={area}, Center=({cx},{cy})")

        debug_pages.append(Image.fromarray(cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)))
        if boxes and not is_fullpage:
            for (x1, y1, x2, y2) in boxes:
                split_pages.append(orig_pil.crop((x1, y1, x2, y2)))
        else:
            split_pages.append(orig_pil)

        if save_debug_output:
            log_detection_results_global(
                pdf_name=filename, page_index=idx,
                boxes=boxes, is_fullpage=is_fullpage,
                is_final=is_final, log_file_path=detection_log_path
            )

        if is_fullpage and show_box_stats:
            print(f"Page {idx+1} marked as FULL PAGE")
        elif show_box_stats:
            print(f"Page {idx+1} contains {len(boxes)} detected box(es)")

    for idx in full_page_indices:
        if len(split_pages) <= idx or not isinstance(split_pages[idx], Image.Image):
            split_pages.insert(idx, orig_pages[idx])
            if show_box_stats:
                print(f"Re-locking Page {idx+1} as FULL PAGE")
        elif show_box_stats:
            print(f"Skipped re-locking Page {idx+1} — already added")

    if save_debug_output:
        debug_pages[0].save(debug_path, save_all=True, append_images=debug_pages[1:], resolution=dpi)
        print(f"Debug saved to: {os.path.basename(debug_path)}")

    split_pages[0].save(split_path, save_all=True, append_images=split_pages[1:], resolution=dpi)
    print(f"Split saved to: {os.path.basename(split_path)}\n")


