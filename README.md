---
title: ResNet Chess Engine
emoji: ♟️
colorFrom: gray
colorTo: blue
sdk: docker
pinned: false
---
# ♟️ MyChessBot

**A Dual-Head SE-ResNet15 engine with 1-ply Batched Tensor Search.**

A neural chess engine that doesn't play *optimal* chess — it plays like *me*. Trained on **3,437 of my own games** (346,000+ positions), this is a behavioral clone that replicates my intuition, pattern recognition, and my specific tactical blind spots.

---

## Architecture

| Component | Detail |
|---|---|
| **Backbone** | SE-ResNet15 — 15 Residual Blocks with Squeeze-and-Excitation attention (192 channels) |
| **Policy Head** | Outputs a 4096-logit probability distribution over all possible moves |
| **Value Head** | Evaluates board positions on a [-1, +1] scale via Tanh activation |
| **Input Encoding** | 19-plane (8×8) tensor: 12 piece planes, 4 castling rights, en passant, side-to-move, temporal encoding |
| **Training Data** | 3,437 of my games → 346,000+ labeled positions with Stockfish evaluations |

## How It Thinks

### 1. Policy Pruning (Instinct)
The Policy Head scores all legal moves and extracts the **Top 5** by raw logit value. No temperature scaling at this stage — rank order is preserved to save compute.

### 2. Tactical Capture Injection
All legal captures are generated via `board.generate_legal_captures()`, sorted by target piece material value (Q > R > B > N > P), deduplicated against the instinct list, and injected into the candidate pool (capped at **10 total**).

### 3. Batched 1-Ply Search
Each candidate move is applied to a cloned board. The resulting future positions are converted to tensors and **stacked into a single `(K, 19, 8, 8)` batch**, passed through the Value Head in one forward call.

### 4. Perspective Negation
The Value Head reports how good the position is *for the side to move* — which, after our move, is the opponent. Each value is negated and converted to centipawns via `atanh(v) × 400` to get *our* advantage.

### 5. Veto-Blend Selection
Moves that fall more than **100 centipawns** below the best candidate are vetoed. Among survivors, temperature-scaled softmax is applied to the original Policy logits, and a move is sampled — preserving my personal style while enforcing tactical safety.

## Limitations

- **1-ply depth only.** The engine cannot see forced mates, pins that resolve in 2+ moves, or deep positional sacrifices.
- **Behavioral clone.** It inherits my personal blind spots. If I never played the Sicilian, neither will the bot.
- **No opening book.** Every move is computed from scratch via the neural network.

## Tech Stack

| Layer | Technology |
|---|---|
| **Model** | PyTorch (CUDA) |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React + Vite + Tailwind CSS |
| **Chess Logic** | python-chess (server) + chess.js (client) |
| **Board UI** | react-chessboard |

## Running Locally

```bash
# Backend
pip install fastapi uvicorn torch numpy python-chess
uvicorn main:app --host 0.0.0.0 --port 8000

# Frontend
cd chess-frontend
npm install
npm run dev
```

## Training

```bash
# 1. Mine your PGN games into tensors
python data_miner.py

# 2. Train the model
python train_model.py
```

The training pipeline uses mixed-precision (FP16), OneCycleLR scheduling, label smoothing (0.1), and early stopping (patience 5). Dual loss: `policy_loss + 5.0 × value_loss`.

---

*Created by [Piyush Singh](https://github.com/singhh-piyush)*
