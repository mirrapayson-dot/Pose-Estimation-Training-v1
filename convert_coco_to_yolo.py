import json
import os
import shutil
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
COCO_JSON      = "annotations/person_keypoints_Train.json"
IMAGES_SRC     = "images/"
OUTPUT_DIR     = "yolo_dataset"
TRAIN_SPLIT    = 0.85

# These must match the category names exactly as they appear in your CVAT export
CAT_PERSON     = "person"
CAT_POLE       = "pole"       # update if you named it differently in CVAT
CAT_ROLLERSKI  = "rollerski"  # update if you named it differently in CVAT

# Combined keypoint count
NUM_KP_PERSON     = 15
NUM_KP_POLE       = 4
NUM_KP_ROLLERSKI  = 4
TOTAL_KP          = NUM_KP_PERSON + NUM_KP_POLE + NUM_KP_ROLLERSKI  # = 23
# ──────────────────────────────────────────────────────────────────────────────


def bbox_iou(b1, b2):
    """Rough IoU overlap between two COCO-format bboxes [x,y,w,h]."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    inter = ix * iy
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def empty_kps(n):
    """Return n invisible zero keypoints as a flat list [x,y,v, x,y,v ...]."""
    return [0.0, 0.0, 0] * n


def extract_kps(ann, n):
    """Pull keypoints from an annotation, padding/truncating to exactly n."""
    kps = ann.get("keypoints", [])
    result = []
    for k in range(n):
        base = k * 3
        if base + 2 < len(kps):
            result.extend([kps[base], kps[base+1], kps[base+2]])
        else:
            result.extend([0.0, 0.0, 0])
    return result


def convert():
    print("convert() started")
    with open(COCO_JSON) as f:
        data = json.load(f)

    # Build category name → id map
    cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    print("Categories found:", list(cat_name_to_id.keys()))

    id_person    = cat_name_to_id.get(CAT_PERSON)
    id_pole      = cat_name_to_id.get(CAT_POLE)
    id_rollerski = cat_name_to_id.get(CAT_ROLLERSKI)

    if not id_person:
        raise ValueError(f"Could not find category '{CAT_PERSON}' in JSON. "
                         f"Available: {list(cat_name_to_id.keys())}")

    images = {img["id"]: img for img in data["images"]}

    # Group annotations by image, then by category
    anns_by_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        anns_by_image.setdefault(img_id, {}).setdefault(cat_id, []).append(ann)

    # Create output dirs
    for split in ("train", "val"):
        Path(f"{OUTPUT_DIR}/images/{split}").mkdir(parents=True, exist_ok=True)
        Path(f"{OUTPUT_DIR}/labels/{split}").mkdir(parents=True, exist_ok=True)

    image_ids = list(images.keys())
    split_idx = int(len(image_ids) * TRAIN_SPLIT)

    for idx, img_id in enumerate(image_ids):
        img_info = images[img_id]
        W, H     = img_info["width"], img_info["height"]
        fname    = img_info["file_name"]
        split    = "train" if idx < split_idx else "val"

        # Copy image
        src = Path(IMAGES_SRC) / fname
        dst = Path(f"{OUTPUT_DIR}/images/{split}") / Path(fname).name
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"  WARNING: image not found: {src}")

        frame_anns  = anns_by_image.get(img_id, {})
        person_anns = frame_anns.get(id_person, [])
        pole_anns   = frame_anns.get(id_pole, []) if id_pole else []
        ski_anns    = frame_anns.get(id_rollerski, []) if id_rollerski else []

        lines = []
        for person in person_anns:
            # ── Bounding box ──────────────────────────────────────────────────
            bx, by, bw, bh = person["bbox"]
            cx = (bx + bw / 2) / W
            cy = (by + bh / 2) / H
            bw_n = bw / W
            bh_n = bh / H

            # ── Person keypoints ─────────────────────────────────────────────
            person_kps = extract_kps(person, NUM_KP_PERSON)

            # ── Match pole annotation to this person by best bbox overlap ─────
            best_pole = None
            best_pole_iou = 0.0
            for pole in pole_anns:
                iou = bbox_iou(person["bbox"], pole["bbox"])
                if iou > best_pole_iou:
                    best_pole_iou = iou
                    best_pole = pole
            pole_kps = extract_kps(best_pole, NUM_KP_POLE) if best_pole else empty_kps(NUM_KP_POLE)

            # ── Match ski annotation to this person by best bbox overlap ──────
            best_ski = None
            best_ski_iou = 0.0
            for ski in ski_anns:
                iou = bbox_iou(person["bbox"], ski["bbox"])
                if iou > best_ski_iou:
                    best_ski_iou = iou
                    best_ski = ski
            ski_kps = extract_kps(best_ski, NUM_KP_ROLLERSKI) if best_ski else empty_kps(NUM_KP_ROLLERSKI)

            # ── Normalize all keypoints ───────────────────────────────────────
            all_kps = person_kps + pole_kps + ski_kps  # flat list, length = 23*3 = 69
            norm_kps = []
            for k in range(TOTAL_KP):
                base = k * 3
                kx  = all_kps[base]     / W
                ky  = all_kps[base + 1] / H
                vis = all_kps[base + 2]
                norm_kps.extend([f"{kx:.6f}", f"{ky:.6f}", str(int(vis))])

            line = f"0 {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f} " + " ".join(norm_kps)
            lines.append(line)

        label_path = Path(f"{OUTPUT_DIR}/labels/{split}") / (Path(fname).stem + ".txt")
        label_path.write_text("\n".join(lines))

    print(f"\nDone. Dataset written to: {OUTPUT_DIR}/")
    print(f"  Train images: {split_idx}")
    print(f"  Val images:   {len(image_ids) - split_idx}")


convert()