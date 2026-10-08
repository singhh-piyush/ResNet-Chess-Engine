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

**Current model:** the previous fold-0 checkpoint is retained as a baseline. It uses the old 4096-action policy; the server explicitly prefers queen promotion and evaluates all promotion pieces independently. The new training vocabulary has 4272 distinct actions. A retrained model is promoted only after independent evaluation. Neural scores are estimates, not Stockfish evaluations.

## Project layout

- `engine/`: shared model, features, promotion encoding, bounded neural search.
- `backend/`: API, validation, readiness, streaming, request concurrency.
- `training/`: public archive download, deterministic offline labels, frozen game splits, GPU training.
- `evaluation/`: independent heldout comparison and release gate.
- `models/release.json`: versioned checkpoint path, architecture and checksum. Weights live in HF, outside GitHub.
- `chess-frontend/`: existing interface source; only backend integration changes in this phase.
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

`GET /health` checks the process. `GET /ready` checks the model and reports its version. `POST /predict` returns a move; `POST /predict/stream` emits real `started`, `progress`, `result` and `error` events. Requests contain `fen`, optionally `starting_fen` and UCI `moves` for repetition-aware decisions. Invalid positions return 422, unavailable models 503, and concurrent searches 429. One search runs at a time; search is capped at 9.5 seconds, four main plies, four quiescence plies and 4096 positions. Model inference is bounded but not preemptible; actual warm latency is verified during deployment.

Set `MODEL_MANIFEST` to select a release and `CORS_ORIGINS` for an external interface. The default allows localhost development; same-origin hosting needs no cross-origin permission.

## Retrain and evaluate

Stockfish is only an offline teacher and evaluator. It is never in the runtime image. Use local CUDA PyTorch and install `requirements-training.txt`.

```bash
venv/bin/python -m training.download
venv/bin/python -m training.mine --workers 4
venv/bin/python -m training.split
venv/bin/python -m training.train --batch 256
# Resume an interrupted run with the identical dataset:
venv/bin/python -m training.train --batch 256 --resume
```

All standard games and time controls are included and deduplicated. Labels use 20,000 Stockfish nodes, one engine thread per worker, no reflected boards, value targets on both turns, and policy targets only on the user's moves losing at most 150 cp. Every row includes game provenance. Mining is atomic per game and resumable. `data/split.json` is frozen and preserves the legacy fold-0 holdout when its archived game IDs are available.

Create a candidate manifest beside its checkpoint with the format in `models/release.json`, `policy_size: 4272`, and the correct checksum. Run:

```bash
venv/bin/python -m evaluation.compare --candidate models/candidate.json
```

At least 500 frozen, unseen positions must show 20% fewer moves losing more than 150 cp, with style top-1 declining at most five percentage points and warm p95 latency under ten seconds. Failed candidates remain research artifacts. Upload verified weights before changing the release manifest; retain the previous checkpoint for rollback.

## Deployment

The existing free CPU Space stays the host. There is no Vercel service in this phase. Chrome is used to configure the HF trusted GitHub publisher, restricted to this repository, branch `main`, workflow `.github/workflows/deploy.yml`. The workflow verifies the exact checkpoint, runs backend tests and builds the interface before syncing application files. It preserves all files under `models/`; old source and bundles are removed only from the current deployed revision and remain recoverable in history. See `docs/status.md` for the current work and verification status.
