"""Convert MVTec D2S to YOLO format shared by ultralytics and rfdetr.

Output (default data/d2s_yolo/):
  c60/, c1/            one dataset per label mode: 60 classes / one class "product"
    data.yaml
    {train,valid,test}/{images,labels}/
  test_a_gt.json       COCO ground truth of test A with per-object `occlusion` and `truncated`

Splits: D2S train -> train/ + valid/ (rotations 4 and 9 of every scene, monitoring only),
D2S val -> test/ (test A). Images are resized to 1024 px on the long side; c1 images are
hard links to the c60 ones. Occlusion comes from D2S Amodal: 1 - visible / amodal area
inside the frame; the amodal share outside the frame decides `truncated`.

Usage: uv run python scripts/prepare_d2s.py [--d2s data/d2s] [--amodal data/d2s_amodal] [--out data/d2s_yolo]
"""

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as M

Coco = dict[str, Any]  # a parsed COCO json file
Ann = dict[str, Any]  # one COCO annotation; D2S masks are compressed RLE that pycocotools reads as-is

W, H = 1920, 1440  # every D2S image
SCALE = 1024 / W
VAL_ROTATIONS = {4, 9}  # image id = scene * 100 + k, rotation = k % 10, lighting = k // 10
TRUNCATED_ABOVE = 0.01  # share of the amodal mask outside the frame


def load(path: Path) -> Coco:
    return json.loads(path.read_text())


def clip(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    assert x1 > x0 and y1 > y0, bbox
    return [x0, y0, x1 - x0, y1 - y0]


def yolo_box(bbox: list[float]) -> str:
    """COCO [x, y, w, h] in pixels -> normalized 'cx cy w h', checked by converting back."""
    x, y, w, h = bbox
    box = f"{(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"
    cx, cy, nw, nh = map(float, box.split())
    back = (cx * W - nw * W / 2, cy * H - nh * H / 2, nw * W, nh * H)
    assert max(abs(a - b) for a, b in zip(back, bbox, strict=True)) < 1, (bbox, box)
    return box


def write_image(src: Path, dst: Path, link: Path) -> None:
    img = cv2.imread(str(src))
    assert img is not None and img.shape[:2] == (H, W), src
    small = cv2.resize(img, (round(W * SCALE), round(H * SCALE)), interpolation=cv2.INTER_AREA)
    assert cv2.imwrite(str(dst), small, [cv2.IMWRITE_JPEG_QUALITY, 95]), dst
    os.link(dst, link)


def padding_offset(d2s_anns: list[Ann], amodal_anns: list[Ann]) -> tuple[int, int]:
    """Where the original frame sits inside a zero-padded amodal image, voted by matching visible-mask boxes."""
    votes: collections.Counter[tuple[int, int]] = collections.Counter()
    for an in amodal_anns:
        x, y, bw, bh = M.toBbox(an["visible_mask"])
        for d in d2s_anns:
            dx, dy, dw, dh = d["bbox"]
            if d["category_id"] == an["category_id"] and abs(bw - dw) <= 1 and abs(bh - dh) <= 1:
                votes[(round(x - dx), round(y - dy))] += 1
    assert votes, "no matching boxes to find the padding offset"
    return votes.most_common(1)[0][0]


def occlusion_by_annotation(val: Coco, amodal: Coco) -> dict[int, tuple[float, bool]]:
    """Match D2S Amodal objects to D2S val objects by visible-mask IoU: {d2s ann id: (occlusion, truncated)}."""
    amodal_imgs = {i["id"]: i for i in amodal["images"]}
    assert set(amodal_imgs) == {i["id"] for i in val["images"]}
    by_d: collections.defaultdict[int, list[Ann]] = collections.defaultdict(list)
    by_a: collections.defaultdict[int, list[Ann]] = collections.defaultdict(list)
    for an in val["annotations"]:
        by_d[an["image_id"]].append(an)
    for an in amodal["annotations"]:
        by_a[an["image_id"]].append(an)

    out: dict[int, tuple[float, bool]] = {}
    for iid, amodal_anns in by_a.items():
        d2s_anns = by_d[iid]
        h, w = amodal_imgs[iid]["height"], amodal_imgs[iid]["width"]
        ox, oy = (0, 0) if (h, w) == (H, W) else padding_offset(d2s_anns, amodal_anns)
        d2s_rles = [d["segmentation"] for d in d2s_anns]
        for an in amodal_anns:
            full = M.decode(an["segmentation"])
            visible = np.asfortranarray(M.decode(an["visible_mask"])[oy : oy + H, ox : ox + W])
            iou = M.iou([M.encode(visible)], d2s_rles, [0] * len(d2s_rles))[0]
            j = int(iou.argmax())
            assert iou[j] > 0.9 and d2s_anns[j]["id"] not in out, (iid, an["id"], iou[j])
            inside = full[oy : oy + H, ox : ox + W].sum()
            out[d2s_anns[j]["id"]] = (
                round(1 - visible.sum() / inside, 4),
                bool(1 - inside / full.sum() > TRUNCATED_ABOVE),
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d2s", type=Path, default=Path("data/d2s"))
    ap.add_argument("--amodal", type=Path, default=Path("data/d2s_amodal"))
    ap.add_argument("--out", type=Path, default=Path("data/d2s_yolo"))
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists, remove it first")

    train = load(args.d2s / "annotations/D2S_training.json")
    val = load(args.d2s / "annotations/D2S_validation.json")
    amodal = load(args.amodal / "annotations/D2S_amodal_validation.json")

    cats = sorted(train["categories"], key=lambda c: c["id"])  # files list categories out of id order
    assert [c["id"] for c in cats] == list(range(1, 61))
    assert sorted(val["categories"], key=lambda c: c["id"]) == cats
    assert not {i["id"] for i in train["images"]} & {i["id"] for i in val["images"]}

    anns: collections.defaultdict[int, list[Ann]] = collections.defaultdict(list)
    for an in train["annotations"] + val["annotations"]:
        assert not an["iscrowd"]
        anns[an["image_id"]].append(an)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": val["images"]}
    for img in train["images"]:
        splits["valid" if img["id"] % 10 in VAL_ROTATIONS else "train"].append(img)

    jobs: list[tuple[Path, Path, Path]] = []
    for split, imgs in splits.items():
        for mode in ("c60", "c1"):
            for sub in ("images", "labels"):
                (args.out / mode / split / sub).mkdir(parents=True)
        for img in imgs:
            boxes = [(an["category_id"] - 1, yolo_box(clip(an["bbox"]))) for an in anns[img["id"]]]
            label = Path(img["file_name"]).with_suffix(".txt").name
            (args.out / "c60" / split / "labels" / label).write_text("".join(f"{c} {b}\n" for c, b in boxes))
            (args.out / "c1" / split / "labels" / label).write_text("".join(f"0 {b}\n" for _, b in boxes))
            c60_img, c1_img = (args.out / mode / split / "images" / img["file_name"] for mode in ("c60", "c1"))
            jobs.append((args.d2s / "images" / img["file_name"], c60_img, c1_img))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda job: write_image(*job), jobs))

    for mode, names in (("c60", [c["name"] for c in cats]), ("c1", ["product"])):
        (args.out / mode / "data.yaml").write_text(
            f"train: train/images\nval: valid/images\ntest: test/images\nnc: {len(names)}\nnames: {json.dumps(names)}\n"
        )

    print("matching D2S Amodal to D2S val (a few minutes)...")
    occ = occlusion_by_annotation(val, amodal)
    unmatched = len(val["annotations"]) - len(occ)
    assert unmatched < 0.01 * len(val["annotations"]), unmatched
    gt = {
        "categories": cats,
        "images": [
            {
                "id": i["id"],
                "file_name": i["file_name"],
                "width": round(W * SCALE),
                "height": round(H * SCALE),
                "scene_id": i["id"] // 100,
            }
            for i in val["images"]
        ],
        "annotations": [
            {
                "id": an["id"],
                "image_id": an["image_id"],
                "category_id": an["category_id"],
                "bbox": [round(v * SCALE, 2) for v in clip(an["bbox"])],
                "area": round(an["area"] * SCALE**2, 1),
                "iscrowd": 0,
                "occlusion": occ.get(an["id"], (None, None))[0],  # None: no amodal annotation
                "truncated": occ.get(an["id"], (None, None))[1],
            }
            for an in val["annotations"]
        ],
    }
    (args.out / "test_a_gt.json").write_text(json.dumps(gt))

    for mode in ("c60", "c1"):
        for split, imgs in splits.items():
            n_img = len(os.listdir(args.out / mode / split / "images"))
            n_lbl = len(os.listdir(args.out / mode / split / "labels"))
            assert n_img == n_lbl == len(imgs), (mode, split, n_img, n_lbl, len(imgs))
    train_labels = (args.out / "c60/train/labels").iterdir()
    train_classes = {int(line.split()[0]) for f in train_labels for line in f.read_text().splitlines()}
    assert train_classes == set(range(60)), sorted(set(range(60)) - train_classes)

    for split, imgs in splits.items():
        print(f"{split:5}: {len(imgs)} images, {sum(len(anns[i['id']]) for i in imgs)} objects")
    o = np.array([a["occlusion"] for a in gt["annotations"] if a["truncated"] is False])
    bins = {
        "0%": o == 0,
        "0-20%": (o > 0) & (o <= 0.2),
        "20-40%": (o > 0.2) & (o <= 0.4),
        "40-60%": (o > 0.4) & (o <= 0.6),
        "60-80%": (o > 0.6) & (o <= 0.8),
        "80-100%": o > 0.8,
    }
    print("test A occlusion, non-truncated objects:", {k: int(v.sum()) for k, v in bins.items()})
    truncated = sum(a["truncated"] is True for a in gt["annotations"])
    print(f"test A: {truncated} truncated, {unmatched} without amodal annotation")


if __name__ == "__main__":
    main()
