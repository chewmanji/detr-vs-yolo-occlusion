"""Predict test A with the final weights of a run and save COCO results JSON for evaluate.py.

Writes <run dir>/pred_test_a.json: a list of {image_id, category_id, bbox [x, y, w, h], score} in the pixel
coordinates of test_a_gt.json, the top 100 boxes per image with score >= 0.001. category_id = class index + 1,
so c1 runs always give 1. YOLO26 runs also write pred_test_a_nms.json: the same weights through the
one-to-many head with NMS (pred_test_a.json is the end-to-end head without NMS). Ultralytics NMS uses its
defaults: IoU 0.7, per class.

Usage: uv run python scripts/predict.py runs/yolo26s_R1_c60_s0 [--data data/d2s_yolo]
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

IMGSZ = 640
CONF, MAX_DET = 0.001, 100
BATCH = 16


def to_coco(image_id: int, xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> list[dict[str, Any]]:
    top = np.argsort(-scores, kind="stable")[:MAX_DET]  # rfdetr always returns 300 boxes
    return [
        {
            "image_id": image_id,
            "category_id": int(classes[i]) + 1,
            "bbox": [round(float(v), 2) for v in (xyxy[i, 0], xyxy[i, 1], *(xyxy[i, 2:] - xyxy[i, :2]))],
            "score": round(float(scores[i]), 5),
        }
        for i in top
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="a run folder written by train.py")
    ap.add_argument("--data", type=Path, default=Path("data/d2s_yolo"))
    args = ap.parse_args()

    gt = json.loads((args.data / "test_a_gt.json").read_text())
    ids = {i["file_name"]: i["id"] for i in gt["images"]}
    image_dir = args.data / "c60/test/images"  # c1 has the same images
    images = [str(image_dir / name) for name in ids]
    model = args.run_dir.name.split("_")[0]
    outputs: dict[str, list[dict[str, Any]]] = {}

    if model == "rfdetrs":
        from rfdetr import from_checkpoint
        from supervision import Detections

        detr = from_checkpoint(args.run_dir / "last_ema.pth")
        preds = []
        for i in range(0, len(images), BATCH):
            batch: list[Any] = images[i : i + BATCH]
            dets = detr.predict(batch, threshold=CONF, include_source_image=False)
            assert isinstance(dets, list)
            for path, d in zip(batch, dets, strict=True):
                assert isinstance(d, Detections) and d.confidence is not None and d.class_id is not None
                preds += to_coco(ids[Path(path).name], d.xyxy, d.confidence, d.class_id)
        outputs["pred_test_a.json"] = preds
    else:
        from ultralytics import YOLO
        from ultralytics.engine.results import Results

        assert model in {"yolo11s", "yolo26s"}, args.run_dir
        yolo = YOLO(args.run_dir / "weights/last.pt")
        # nms=False selects YOLO26's end-to-end head; None and True its one-to-many head with NMS
        variants = {"pred_test_a.json": False, "pred_test_a_nms.json": True} if model == "yolo26s" else {}
        for name, nms in (variants or {"pred_test_a.json": None}).items():
            preds = []
            # a folder, not the list of paths: ultralytics loads a list into memory as one batch
            for r in yolo.predict(
                str(image_dir),
                imgsz=IMGSZ,
                conf=CONF,
                max_det=MAX_DET,
                nms=nms,
                batch=BATCH,
                stream=True,
                verbose=False,
            ):
                assert isinstance(r, Results) and r.boxes is not None
                b = r.boxes.cpu().numpy()
                xyxy, scores, classes = (np.asarray(t) for t in (b.xyxy, b.conf, b.cls))
                preds += to_coco(ids[Path(r.path).name], xyxy, scores, classes)
            outputs[name] = preds

    for name, preds in outputs.items():
        per_image = np.bincount([p["image_id"] for p in preds])
        assert {p["image_id"] for p in preds} <= set(ids.values()) and per_image.max() <= MAX_DET
        (args.run_dir / name).write_text(json.dumps(preds))
        print(f"{args.run_dir / name}: {len(preds)} boxes on {np.count_nonzero(per_image)} of {len(ids)} images")


if __name__ == "__main__":
    main()
