"""Train one run of the experiment matrix with its package's default recipe at 640 px.

Run name: {model}_{regime}_{labels}_s{seed}, e.g. rfdetrs_R2_c60_s1, saved in --out/<run name>/.
Both packages save a checkpoint every epoch (ultralytics weights/last.pt, rfdetr last.ckpt); rerunning the
same command resumes from it, e.g. in a new Kaggle session after copying the previous output back to --out.
Final weights for testing: ultralytics weights/last.pt, rfdetr last_ema.pth.

Datasets: R1 = --data/<labels>, R2 = --data/<labels>_r2. R2 trains for --epochs scaled by the ratio of
training images, so both regimes see the same number of images (Plan.md, decision 8).

Each finished session appends a row to --out/runs.csv: versions, GPU, epochs, wall time, peak GPU memory and
the last validation mAP. A resumed run adds one row per session, so its GPU hours are the sum of its rows.
s_per_epoch is session time / session epochs, so it includes setup and overestimates short runs.

Usage: uv run python scripts/train.py --model rfdetr-s --regime R1 --labels c60 --seed 0 --epochs 3
"""

import argparse
import csv
import os
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch

IMGSZ = 640
RFDETR_BATCH, RFDETR_ACCUM = 4, 4  # effective batch 16, the same as the ultralytics default
# per-epoch log of each package: file, mAP@50:95 and mAP@50 columns, number of the first epoch
LOGS = {
    "rfdetr": ("metrics.csv", "val/mAP_50_95", "val/mAP_50", 0),
    "ultralytics": ("results.csv", "metrics/mAP50-95(B)", "metrics/mAP50(B)", 1),
}


def last_validation(run_dir: Path, package: str) -> tuple[int, dict[str, str]]:
    """Epochs done and the last validation row of the package's per-epoch log; (0, {}) before the first epoch."""
    file, map_col, _, first = LOGS[package]
    log = run_dir / file
    rows = [r for r in csv.DictReader(log.read_text().splitlines()) if r[map_col]] if log.exists() else []
    return (int(rows[-1]["epoch"]) + 1 - first, rows[-1]) if rows else (0, {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["yolo11s", "yolo26s", "rfdetr-s"])
    ap.add_argument("--regime", required=True, choices=["R1", "R2"])
    ap.add_argument("--labels", required=True, choices=["c60", "c1"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--epochs", required=True, type=int, help="R1 epochs; R2 gets the same number of images")
    ap.add_argument("--data", type=Path, default=Path("data/d2s_yolo"))
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    r1 = args.data / args.labels
    dataset = r1 if args.regime == "R1" else args.data / f"{args.labels}_r2"
    assert (dataset / "data.yaml").exists(), f"{dataset} has no data.yaml"
    epochs = args.epochs
    if args.regime == "R2":
        n_r1, n_r2 = (len(os.listdir(d / "train/images")) for d in (r1, dataset))
        epochs = round(args.epochs * n_r1 / n_r2)
    out = args.out.resolve()  # ultralytics puts a relative project under its own runs dir
    run = f"{args.model.replace('-', '')}_{args.regime}_{args.labels}_s{args.seed}"
    run_dir = out / run
    package = "rfdetr" if args.model == "rfdetr-s" else "ultralytics"
    epochs_before, _ = last_validation(run_dir, package)
    started, t0 = datetime.now(), time.time()
    print(f"{run}: {epochs} epochs on {dataset}")

    if package == "rfdetr":
        from rfdetr import RFDETRSmall

        last = run_dir / "last.ckpt"
        RFDETRSmall().train(
            dataset_dir=str(dataset),
            output_dir=str(run_dir),
            epochs=epochs,
            resolution=IMGSZ,  # must divide by patch 16 x 2 windows
            batch_size=RFDETR_BATCH,
            grad_accum_steps=RFDETR_ACCUM,
            seed=args.seed,
            resume=str(last) if last.exists() else None,
        )
    else:
        from ultralytics import YOLO

        last = run_dir / "weights/last.pt"
        YOLO(last if last.exists() else f"{args.model}.pt").train(
            data=str(dataset / "data.yaml"),
            project=str(out),
            name=run,
            exist_ok=True,  # a run that crashed before its first checkpoint restarts in the same folder
            epochs=epochs,
            imgsz=IMGSZ,
            seed=args.seed,
            deterministic=True,
            resume=str(last) if last.exists() else False,  # a finished run refuses to resume
        )

    hours = (time.time() - t0) / 3600
    epochs_done, row = last_validation(run_dir, package)
    session_epochs = epochs_done - epochs_before
    _, map_col, map50_col, _ = LOGS[package]
    cuda = torch.cuda.is_available()
    summary = {
        "run": run,
        "started": started.isoformat(timespec="minutes"),
        "gpu": torch.cuda.get_device_name() if cuda else "cpu",
        "torch": torch.__version__,
        "ultralytics": version("ultralytics"),
        "rfdetr": version("rfdetr"),
        "epochs": epochs_done,
        "session_epochs": session_epochs,
        "hours": round(hours, 3),
        "s_per_epoch": round(hours * 3600 / session_epochs) if session_epochs else "",
        "peak_gpu_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2) if cuda else "",
        "val_map50_95": row[map_col],
        "val_map50": row[map50_col],
    }
    log = out / "runs.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            writer.writeheader()
        writer.writerow(summary)
    print(summary)


if __name__ == "__main__":
    main()
