"""Evaluate predictions on test A: COCO mAP and recall per occlusion bin (Plan.md, section 7).

Input: pred_test_a*.json files that predict.py writes into run folders; the folder name gives the run and its
label mode. Columns per file:
  nms              whether boxes went through NMS: YOLO11, and YOLO26 in pred_test_a_nms.json
  map50_95, map50  pycocotools at maxDets 100; class-aware for c60, class-agnostic for c1
  any_<group>      Recall_any: share of objects matched by a box of any class
  cls_<group>      Recall_cls: the same with boxes of the right class only (c60 runs)
Matching is greedy by descending score: a box takes the unmatched object it overlaps most with IoU >= 0.5, so
one box finds at most one object. Every saved box counts (score >= 0.001), so this is recall without a
confidence threshold; the operating point at precision 0.90 is added in stage 5.
Groups: all objects; non-truncated objects by occlusion 0-20/20-40/40-60/60-80/80-100% (a bin includes its
upper edge, 0% is in 0-20); truncated objects. Objects without an amodal annotation are only in "all".

Usage: uv run python scripts/evaluate.py RUN_DIR/pred_test_a.json ... [--gt data/d2s_yolo/test_a_gt.json]
       [--out results/test_a.csv]
"""

import argparse
import collections
import contextlib
import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as M
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IOU = 0.5
BINS = {"0-20": 0.2, "20-40": 0.4, "40-60": 0.6, "60-80": 0.8, "80-100": 1.0}  # upper edges
RUN = re.compile(r"(yolo11s|yolo26s|rfdetrs)_(R1|R2)_(c60|c1)_s(\d+)")


def match(
    gt: np.ndarray, gt_cls: np.ndarray, dt: np.ndarray, dt_cls: np.ndarray, scores: np.ndarray, same_class: bool
) -> np.ndarray:
    """Which objects a greedy matching by descending score finds; boxes are COCO [x, y, w, h]."""
    hit = np.zeros(len(gt), bool)
    if len(gt) and len(dt):
        ious = np.asarray(M.iou(dt, gt, [0] * len(gt)))  # (boxes, objects)
        for d in np.argsort(-scores, kind="stable"):
            free = ~hit & (ious[d] >= IOU) & ((gt_cls == dt_cls[d]) | (not same_class))
            if free.any():
                hit[np.flatnonzero(free)[ious[d, free].argmax()]] = True
    return hit


def check_match() -> None:
    two = np.array([[0, 0, 10, 10], [5, 0, 10, 10]], float)  # two objects overlapping with IoU 1/3
    assert match(two, np.array([1, 1]), two[:1], np.array([1]), np.array([0.9]), False).tolist() == [True, False]
    cls = two, np.array([1, 2]), two, np.array([2, 2]), np.array([0.9, 0.8])
    assert match(*cls, same_class=False).tolist() == [True, True]
    assert match(*cls, same_class=True).tolist() == [False, True]  # the box on object 1 has the wrong class


def coco_map(gt: dict[str, Any], preds: list[dict[str, Any]], agnostic: bool) -> tuple[float, float]:
    """mAP@50:95 and mAP@50 from pycocotools; agnostic maps every class to one."""
    if agnostic:
        gt = {
            **gt,
            "categories": [{"id": 1, "name": "product"}],
            "annotations": [{**a, "category_id": 1} for a in gt["annotations"]],
        }
    preds = [{**p, "category_id": 1} if agnostic else dict(p) for p in preds]  # loadRes edits its input
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress and the summary table
        coco = COCO()
        coco.dataset = gt  # pyright: ignore[reportAttributeAccessIssue]  # the stub types it as a TypedDict
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes(preds), "bbox")  # pyright: ignore[reportArgumentType]  # takes a list too
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def groups(anns: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Boolean mask over the objects for every reported group."""
    occ = np.array([np.nan if a["occlusion"] is None else a["occlusion"] for a in anns])
    truncated = np.array([a["truncated"] is True for a in anns])
    clean = ~truncated & ~np.isnan(occ)
    bin_of = np.searchsorted(list(BINS.values()), occ)  # first upper edge >= occlusion
    return (
        {"all": np.ones(len(anns), bool)}
        | {g: clean & (bin_of == i) for i, g in enumerate(BINS)}
        | {"truncated": truncated}
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preds", type=Path, nargs="+", help="pred_test_a*.json files in run folders")
    ap.add_argument("--gt", type=Path, default=Path("data/d2s_yolo/test_a_gt.json"))
    ap.add_argument("--out", type=Path, default=Path("results/test_a.csv"))
    args = ap.parse_args()
    check_match()

    gt = json.loads(args.gt.read_text())
    anns = gt["annotations"]
    masks = groups(anns)
    gt_box = np.array([a["bbox"] for a in anns], float)
    gt_cls = np.array([a["category_id"] for a in anns])
    objects: collections.defaultdict[int, list[int]] = collections.defaultdict(list)
    for i, a in enumerate(anns):
        objects[a["image_id"]].append(i)

    rows = []
    for path in args.preds:
        run = RUN.fullmatch(path.parent.name)
        assert run, f"{path.parent.name} is not a run folder of train.py"
        model, regime, labels, seed = run.groups()
        agnostic = labels == "c1"
        preds = json.loads(path.read_text())
        boxes: collections.defaultdict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for p in preds:
            boxes[p["image_id"]].append(p)
        hit_any, hit_cls = np.zeros(len(anns), bool), np.zeros(len(anns), bool)
        for image_id, idx in objects.items():
            ds = boxes[image_id]
            dt = np.array([d["bbox"] for d in ds], float).reshape(-1, 4)
            dt_cls, scores = np.array([d["category_id"] for d in ds]), np.array([d["score"] for d in ds])
            hit_any[idx] = match(gt_box[idx], gt_cls[idx], dt, dt_cls, scores, same_class=False)
            hit_cls[idx] = match(gt_box[idx], gt_cls[idx], dt, dt_cls, scores, same_class=True)
        map50_95, map50 = coco_map(gt, preds, agnostic)
        rows.append(
            {
                "run": path.parent.name,
                "nms": model == "yolo11s" or path.stem.endswith("_nms"),
                "model": model,
                "regime": regime,
                "labels": labels,
                "seed": int(seed),
                "map50_95": round(map50_95, 4),
                "map50": round(map50, 4),
            }
            | {f"any_{g}": round(float(hit_any[m].mean()), 4) for g, m in masks.items()}
            | {f"cls_{g}": "" if agnostic else round(float(hit_cls[m].mean()), 4) for g, m in masks.items()}
        )

    print("objects per group:", {g: int(m.sum()) for g, m in masks.items()})
    print(f"{'run':22} {'nms':5} {'mAP50-95':>8} {'mAP50':>6}  recall_any (recall_cls): " + " ".join(masks))
    for r in rows:
        recall = [f"{r[f'any_{g}']:.3f}" + (f" ({r[f'cls_{g}']:.3f})" if r["labels"] == "c60" else "") for g in masks]
        print(f"{r['run']:22} {r['nms']!s:5} {r['map50_95']:8.3f} {r['map50']:6.3f}  " + "  ".join(recall))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
