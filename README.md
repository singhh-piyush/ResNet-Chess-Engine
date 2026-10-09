---
title: ResNet Chess Engine
emoji: ♟️
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Personal chess engine

A neural chess engine that learns `piyushhsingh`'s style while searching for safer moves. Hugging Face runs the CPU backend and builds the existing interface from source. GitHub is the source of truth; tested `main` changes deploy to the existing Space using scoped, short-lived credentials.

**Current model:** `candidate-v2`, the 15-block network fine-tuned from the legacy fold-0 checkpoint on the 4272-action vocabulary, served as ONNX with an opening book of my most common recent moves. On test games that no model ever trained on, it matches my move 43.3% of the time, against 42.5% for the legacy checkpoint. Its value error is 0.111, against 0.139. Typical moves take about a second on two CPU threads, and the hard cap is 3 seconds. The legacy checkpoint stays on the Space for rollback (`models/legacy-cv0.json`). Neural scores are estimates, not Stockfish evaluations.

**Known evaluation issue:** `legacy-cv0` was trained on 2,760 of 3,449 games. Its true held-out fold is sklearn's `GroupKFold` fold 0 *with stable tie-breaking*; that reproduces its logged 42.66% validation top-1 exactly. The installed library breaks ties differently, so `data/split.json` held out the wrong games, and 262 of the 331 test games are ones the legacy model trained on. Any model initialized from it scores 78–81% on those games but about 43% on unseen ones. Earlier "73%" figures, and the K-fold numbers for legacy-initialized models, measure memorization. Honest comparisons use `data/evaluation/suite-clean.jsonl`: 1,600 positions from the 69 test games no model has seen.

## Project layout

- `engine/`: shared model, features, promotion encoding, bounded neural search.
- `backend/`: API, validation, readiness, streaming, request concurrency.
- `training/`: public archive download, deterministic offline labels, frozen game splits, GPU training.
- `evaluation/`: independent heldout comparison and release gate.
- `models/release.json`: versioned checkpoint path, architecture and checksum. Weights live in HF, outside GitHub.
- `chess-frontend/`: React + Vite interface, built into the Space image.
- `tests/`, `scripts/`, `.github/`: verification and deployment.
- `archive/`, `data/`, `venv/`: ignored local legacy files, research artifacts and environment. The archive preserves original files and uncommitted training work.

## Run the backend

Python 3.11 or newer. Install PyTorch for your hardware first. Serving on HF uses CPU-only PyTorch.

```bash
python -m venv venv
venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
venv/bin/python -m pip install -r requirements-dev.txt
```

Download the file specified in `models/release.json` from the Space into `models/`, then run `./run_project.sh`. The backend listens on `http://127.0.0.1:7860`. For interface development run `npm ci` and `npm run dev` inside `chess-frontend`; Vite proxies API calls to the backend.

`GET /health` checks the process. `GET /ready` checks the model and reports its version. `POST /predict` returns a move; `POST /predict/stream` emits real `started`, `progress`, `result` and `error` events. Requests contain `fen`, optionally `starting_fen` and UCI `moves` for repetition-aware decisions. Invalid positions return 422, unavailable models 503, and concurrent searches 429. One search runs at a time. Book positions answer instantly from your own recent games. Otherwise iterative deepening uses a 2-second soft budget (no iteration starts that cannot fit) and a 3-second hard cap (`SEARCH_SECONDS`, `SEARCH_HARD_SECONDS`), with a history-aware transposition table, root windowing that only resolves moves near the best, and batched horizon evaluation. Model inference is bounded but not preemptible; actual warm latency is verified during deployment.

Set `MODEL_MANIFEST` to select a release and `CORS_ORIGINS` for an external interface. The runtime uses CUDA when available and the CPU otherwise (`ENGINE_DEVICE` overrides); manifests with `"format": "onnx"` serve through ONNX Runtime on CPU and fall back to their PyTorch source on a GPU. The default allows localhost development; same-origin hosting needs no cross-origin permission.

## Retrain and evaluate

Stockfish is only an offline teacher and evaluator. It is never in the runtime image. Use local CUDA PyTorch and install `requirements-training.txt`.

```bash
venv/bin/python -m training.download
venv/bin/python -m training.mine --workers 4
venv/bin/python -m training.split
# Everything below resumes after interruption:
venv/bin/python -m scripts.retrain_local
```

The pipeline runs on the GPU wherever possible. `training.pack` parses labels once into `data/packed/` (~230 MB), which training keeps entirely in GPU memory. `training.kfold` splits the train+validation games into five game-level folds (the test split is never read):

1. `--stage sweep` fine-tunes every config in `GRID` from `legacy-cv0` on each fold, choosing by style top-1 on the last 24 months of held-out games. Because `legacy-cv0` already saw most pool games, these fold scores are inflated; all eight configs tied within noise.
2. `--stage teachers` trains the best config per fold; each teacher labels only its held-out fold, giving out-of-fold soft targets.
3. `--stage student` distills those targets plus your moves into a smaller network: `--cv-fold 0` to compare sizes, `--final` to train on the whole pool and write `models/student-<size>.json`. A from-scratch 10x128 student reached 39.8% top-1 on its held-out fold, which is honest. The teachers' targets add little, because they memorized the pool through `legacy-cv0`.

`training.book` builds the opening book from train+validation games; `scripts.export_onnx` exports and verifies the ONNX model. `training.train` alone runs one fine-tune on the frozen train/validation split.

All standard games and time controls are included and deduplicated. Labels use 20,000 Stockfish nodes, one engine thread per worker, no reflected boards, value targets on both turns, and policy targets only on the user's moves losing at most 150 cp. Every row includes game provenance. Mining is atomic per game and resumable. `data/split.json` is frozen. It was meant to preserve the legacy fold-0 holdout but does not; see *Known evaluation issue* above.

Evaluate a candidate manifest in two steps:

```bash
venv/bin/python -m evaluation.compare --candidate models/candidate.json --output data/evaluation/<name> --latency-only
venv/bin/python -m evaluation.compare --candidate models/candidate.json --output data/evaluation/<name>
```

The first run measures latency on CPU with two threads (as on the Space) and records how many search positions each model reaches in its budget. The second searches the frozen test positions on the GPU (`--suite data/evaluation/suite-clean.jsonl --count 1600` for the uncontaminated suite) with exactly those position budgets and scores the moves with Stockfish in a cached process pool. To pass:

- blunders must fall: the bootstrap 95% interval of the change must lie below zero, or the rate must drop by at least 15%;
- the lower bound of the style top-1 change must stay within two points;
- candidate p95 latency must be at most three seconds.

Reports include top-3 accuracy, search-move match, ply buckets, the last 12 months and the book-hit rate. Failed candidates remain research artifacts. Upload verified weights before changing the release manifest; retain the previous checkpoint for rollback. To roll back, copy `models/legacy-cv0.json` over `models/release.json` and push.

## Deployment

The existing free CPU Space stays the host. There is no Vercel service in this phase. Chrome is used to configure the HF trusted GitHub publisher, restricted to this repository, branch `main`, workflow `.github/workflows/deploy.yml`. The workflow verifies the exact checkpoint, runs backend tests and builds the interface before syncing application files. It preserves all files under `models/`; old source and bundles are removed only from the current deployed revision and remain recoverable in history. See `docs/status.md` for the current work and verification status.
