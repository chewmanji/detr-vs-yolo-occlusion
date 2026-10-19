# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

No code yet: the repo holds only planning docs. There are no build, lint or test commands. Add them here once code exists.

## Project

A solo academic project (written in Polish) that measures how object detectors degrade under occlusion in crowded self-checkout scenes. `docs/Temat.md` is the source of truth for scope, research questions (RQ1–RQ4), schedule and literature. Read it before proposing any design.

Core comparison (models of similar size, so the comparison is fair):
- **YOLO11-S**: CNN with classic NMS (baseline)
- **YOLO26-S**: CNN, end-to-end without NMS. It isolates the NMS effect. It also has an optional one-to-many head with NMS, so NMS can be compared on a single model.
- **RF-DETR Small**: transformer with a DINOv2 backbone, without NMS. It isolates the architecture effect.

Experiment matrix: 3 models × 2 training regimes (R1 = isolated objects only, R2 = R1 + synthetic copy-paste scenes) × 2 label modes (60 classes / class-agnostic "product") = 12 runs, plus a second seed for the 6 60-class configs = 18 runs.

## Planned stack and data decisions

- PyTorch, `ultralytics` and `rfdetr` packages, Kaggle free GPU, input resolution ~640 px.
- Both packages accept YOLO-format annotations, so the plan relies on **one** D2S → YOLO conversion shared by all models.
- Dataset: MVTec D2S. The D2S train split has isolated objects on a uniform background, and val has crowded scenes. Test annotations are not public, so **D2S val serves as the test set** and validation is split off from train. Fallback dataset: RPC.
- Test A: real occlusion on D2S val, with the occlusion level estimated from box/mask overlap.
- Test B: synthetic scenes built from D2S mask cutouts, where the exact occlusion % is known. Results are binned at 0–20/20–40/40–60/60–80%. Cutouts must come from images **not used in training**.
- Primary metric: recall at IoU 0.5 per occlusion bin (a missed product = an unscanned product). Also COCO mAP@50:95 and mAP@50, the class-confusion share, and CPU latency (ms/img).
- Scope is limited to closed-set detection on the 60 D2S classes. No embeddings, open-set matching or video: those belong to the future master's thesis.
- Licensing: D2S is non-commercial and Ultralytics is AGPL-3.0, so the project stays purely academic.
- GPU budget checkpoint (M1, 2026-11-03): if RF-DETR epoch time makes 18 runs infeasible, drop the second seed.

## Commits

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `<type>[optional scope]: <description>`, imperative, lowercase description, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`, `perf`, `style`. Breaking change: `!` after type/scope or a `BREAKING CHANGE:` footer. Example: `feat(data): add D2S to YOLO converter`.

## Repo notes

- `docs/Przegląd/` (literature PDFs) is gitignored and local only.
- `docs/plan_czytania.md` is the suggested reading order.
