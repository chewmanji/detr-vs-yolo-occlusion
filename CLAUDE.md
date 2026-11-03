# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Data preparation done; training code not written yet. Python 3.11 with dependencies managed by uv (`pyproject.toml` + `uv.lock`):

- `uv sync` installs the locked environment into `.venv`
- `uv run python <script>` runs inside it
- `uv add <pkg>` adds a dependency (`--dev` for tools)
- `uv run ruff check .` lints, `uv run ruff format .` formats and `uv run pyright` type-checks; run all three before committing. Config is in `pyproject.toml`: line length 120, type hints required on every function. Pyright uses the same engine and config as Pylance and checks only `scripts/`; add new code folders to its `include`
- `uv run python scripts/prepare_d2s.py` builds `data/d2s_yolo/` from `data/d2s` and `data/d2s_amodal` (about 5 min): `c60/` and `c1/` YOLO datasets readable by both packages, plus `test_a_gt.json` with per-object `occlusion` and `truncated`. It refuses to overwrite an existing output; its asserts are the checks.

Locally torch is the CPU build, used for data prep, evaluation and CPU latency. Training runs on Kaggle, which keeps its own CUDA torch and installs only the locked `ultralytics` and `rfdetr` versions. There is no test suite yet: scripts check themselves with asserts.

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
- Dataset: MVTec D2S. The D2S train split has isolated objects on a uniform background, and val has crowded scenes. Test annotations are not public, so **D2S val serves as the test set**. Validation is 2 of the 10 rotations of each training scene, used only to monitor training; models are tested on their final weights. Fallback dataset: RPC.
- Data lives in `data/d2s` and `data/d2s_amodal` (gitignored except `.md` files). Image id = scene × 100 + k, rotation = k % 10, lighting = k // 10; never split one scene between training and testing.
- Test A: real occlusion on D2S val, with exact per-object occlusion from D2S Amodal: 1 − visible/amodal area inside the frame. Amodal images are zero-padded, so masks are shifted back to D2S coordinates; objects cut by the image edge are tracked separately. Test A has few objects above 40% occlusion.
- Test B: synthetic scenes built from D2S mask cutouts, where the exact occlusion % is known. Results are binned at 0–20/20–40/40–60/60–80%. Cutouts come from unoccluded objects in D2S val, which no model trains on.
- Primary metric: recall at IoU 0.5 per occlusion bin (a missed product = an unscanned product). Also COCO mAP@50:95 and mAP@50, the class-confusion share, and CPU latency (ms/img).
- Scope is limited to closed-set detection on the 60 D2S classes. No embeddings, open-set matching or video: those belong to the future master's thesis.
- Licensing: D2S is non-commercial and Ultralytics is AGPL-3.0, so the project stays purely academic.
- GPU budget checkpoint (M1, 2026-11-03): if RF-DETR epoch time makes 18 runs infeasible, drop the second seed.

## Commits

Follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/): `<type>[optional scope]: <description>`, imperative, lowercase description, no trailing period. Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `ci`, `perf`, `style`. Breaking change: `!` after type/scope or a `BREAKING CHANGE:` footer. Example: `feat(data): add D2S to YOLO converter`.

## Repo notes

- `docs/Przegląd/` (literature PDFs) is gitignored and local only.
- `docs/Plan.md` is the detailed work plan: stages with task checklists, methodological decisions and open decisions. Tick its checkboxes as tasks get done.
