import fastapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import chess
import math
import os
import contextlib

MODEL_PATH = "chess_clone.pth"
TEMPERATURE = 0.8
K_CANDIDATES = 5
BLUNDER_THRESHOLD = 100
MAX_BATCH = 10           # Cap batch at 10 for CPU inference.
PIECE_VALUE = {'q': 9, 'r': 5, 'b': 3, 'n': 3, 'p': 1}

inference_cache = {}

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop2d = nn.Dropout2d(0.2)
        self.se = SEBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop2d(out)
        out = self.se(out)
        out += residual
        out = self.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, num_res_blocks=15, channels=192):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.res_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(32 * 8 * 8, 4096)
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_tower(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def board_to_tensor(board):
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                layer += 6
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[layer, rank, file] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[17, ep_rank, ep_file] = 1.0

    tensor[18, :, :] = min(board.fullmove_number / 60.0, 1.0)

    return tensor

def encode_move(move):
    return move.from_square * 64 + move.to_square

def decode_move(move_idx):
    from_sq = move_idx // 64
    to_sq = move_idx % 64
    return chess.Move(from_sq, to_sq)

model = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Initializing Backend...")

    if os.path.exists(MODEL_PATH):
        try:
            model = ChessModel()
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"Model not found at {MODEL_PATH}")

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FenRequest(BaseModel):
    fen: str

@app.post("/predict")
async def predict(request: FenRequest):
    global model

    if not model:
        return {"move": None, "error": "Model not loaded"}

    board = chess.Board(request.fen)
    legal_moves_list = list(board.legal_moves)

    if not legal_moves_list:
        return {"move": None, "game_over": True}

    if request.fen in inference_cache:
        cached = inference_cache[request.fen].copy()
        cached["thinking_log"] = ["Cache hit. Returning calculated line."]
        return cached

    thinking_log = []
    thinking_log.append(f"Analyzing position (Move {board.fullmove_number})...")

    tensor = board_to_tensor(board)
    input_tensor = torch.from_numpy(tensor).unsqueeze(0)

    with torch.no_grad():
        raw_logits, _ = model(input_tensor)

    mask = torch.full((1, 4096), float('-inf'))
    legal_moves_map = {}
    for move in legal_moves_list:
        idx = encode_move(move)
        mask[0, idx] = 0.0
        legal_moves_map[idx] = move

    masked_logits = raw_logits + mask

    legal_indices = list(legal_moves_map.keys())
    legal_logits_tensor = masked_logits[0, legal_indices]
    top_k = min(K_CANDIDATES, len(legal_indices))
    top_vals, top_local_indices = torch.topk(legal_logits_tensor, top_k)

    policy_probs = torch.softmax(masked_logits, dim=1)

    top_candidates = []
    for i in range(top_k):
        idx = legal_indices[top_local_indices[i].item()]
        top_candidates.append({
            "move_obj": legal_moves_map[idx],
            "move_idx": idx,
            "raw_logit": top_vals[i].item(),
            "policy_conf": policy_probs[0, idx].item(),
        })

    thinking_log.append(
        f"Policy pruned {len(legal_moves_list)} legal moves to top {top_k} instincts."
    )

    seen_indices = {c["move_idx"] for c in top_candidates}

    captures = list(board.generate_legal_captures())
    captures.sort(
        key=lambda m: PIECE_VALUE.get(
            (board.piece_at(m.to_square).symbol().lower() if board.piece_at(m.to_square) else 'p'), 1
        ),
        reverse=True,
    )

    forced_count = 0
    for cap_move in captures:
        if len(top_candidates) >= MAX_BATCH:
            break
        cap_idx = encode_move(cap_move)
        if cap_idx in seen_indices:
            continue
        seen_indices.add(cap_idx)
        top_candidates.append({
            "move_obj": cap_move,
            "move_idx": cap_idx,
            "raw_logit": masked_logits[0, cap_idx].item(),
            "policy_conf": policy_probs[0, cap_idx].item(),
        })
        forced_count += 1

    if forced_count > 0:
        thinking_log.append(
            f"Tactical forcing: +{forced_count} capture(s) injected. "
            f"Total batch: {len(top_candidates)} candidates."
        )

    virtual_tensors = []
    for cand in top_candidates:
        virtual_board = board.copy()
        virtual_board.push(cand["move_obj"])
        t = board_to_tensor(virtual_board)
        virtual_tensors.append(torch.from_numpy(t))

    batch_tensor = torch.stack(virtual_tensors)

    with torch.no_grad():
        _, virtual_values = model(batch_tensor)

    for i, cand in enumerate(top_candidates):
        raw_val = virtual_values[i].item()
        # atanh scaling logic.
        raw_val = max(min(raw_val, 0.999), -0.999)
        cp = math.atanh(raw_val) * 400.0
        cand["simulated_eval_cp"] = -cp

    best_eval_cp = max(c["simulated_eval_cp"] for c in top_candidates)

    survivors = [
        c for c in top_candidates
        if (best_eval_cp - c["simulated_eval_cp"]) <= BLUNDER_THRESHOLD
    ]

    if not survivors:
        survivors = [max(top_candidates, key=lambda c: c["simulated_eval_cp"])]

    vetoed_count = top_k - len(survivors)
    if vetoed_count > 0:
        thinking_log.append(f"Value Head vetoed {vetoed_count} move(s) (>{BLUNDER_THRESHOLD}cp worse).")

    survivor_logits = torch.tensor(
        [c["raw_logit"] for c in survivors], dtype=torch.float32
    )
    scaled_survivor_logits = survivor_logits / TEMPERATURE
    survivor_probs = torch.softmax(scaled_survivor_logits, dim=0)
    chosen_survivor_idx = torch.multinomial(survivor_probs, 1).item()

    chosen_cand = survivors[chosen_survivor_idx]
    chosen_move = chosen_cand["move_obj"]
    chosen_conf = chosen_cand["policy_conf"]
    chosen_eval_cp = chosen_cand["simulated_eval_cp"]

    standard_eval = round(chosen_eval_cp / 100.0, 2)
    if not board.turn:
        standard_eval = -standard_eval

    thinking_log.append(
        f"1-Ply Veto-Blend complete. {len(survivors)}/{top_k} survived. "
        f"Best: {board.san(chosen_move)} "
        f"(policy {chosen_conf:.1%}, eval {'+' if standard_eval >= 0 else ''}{standard_eval})"
    )
    thinking_log.append(f"Final Decision: {board.san(chosen_move)}")

    survivor_set = {id(c["move_obj"]) for c in survivors}
    candidates_out = []
    for cand in top_candidates:
        eval_pawns = round(cand["simulated_eval_cp"] / 100.0, 2)
        if not board.turn:
            eval_pawns = -eval_pawns
        if cand["move_obj"] == chosen_move:
            status = "SELECTED"
        elif id(cand["move_obj"]) not in survivor_set:
            status = "VETOED"
        else:
            status = "ANALYZED"
        candidates_out.append({
            "move": cand["move_obj"].uci(),
            "san": board.san(cand["move_obj"]),
            "confidence": round(cand["policy_conf"], 4),
            "evaluation": eval_pawns,
            "status": status,
        })

    result = {
        "move": chosen_move.uci(),
        "confidence": round(chosen_conf, 4),
        "evaluation": standard_eval,
        "candidates": candidates_out,
        "thinking_log": thinking_log,
        "blunder_risk": 0.0,
        "agreement": 0.0,
        "is_fallback": False,
    }

    inference_cache[request.fen] = result

    return result

class DrawRequest(BaseModel):
    fen: str
    user_side: str = "white"

@app.post("/offer_draw")
async def offer_draw(request: DrawRequest):
    global model

    if not model:
        return {"accepted": True, "message": "Model unavailable, accepting draw."}

    board = chess.Board(request.fen)
    tensor = board_to_tensor(board)
    input_tensor = torch.from_numpy(tensor).unsqueeze(0)

    with torch.no_grad():
        _, value_out = model(input_tensor)

    raw_val = value_out.item()
    raw_val = max(min(raw_val, 0.999), -0.999)
    cp_eval = math.atanh(raw_val) * 400.0
    standard_eval = cp_eval / 100.0
    if not board.turn:
        standard_eval = -standard_eval

    bot_is_white = request.user_side != "white"
    bot_eval = standard_eval if bot_is_white else -standard_eval

    if bot_eval < -0.5:
        return {"accepted": True, "message": "I accept. Well played."}
    elif bot_eval <= 0.2:
        return {"accepted": True, "message": "Fair enough. Draw agreed."}
    else:
        return {"accepted": False, "message": "I'd like to play on."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
