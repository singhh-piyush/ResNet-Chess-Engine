import requests
import chess
import chess.pgn
import chess.engine
import numpy as np # type: ignore
import math
import json
import os
import io
import time
import multiprocessing
from tqdm import tqdm # type: ignore

# --- Configuration ---
USERNAME = 'piyushhsingh'
STOCKFISH_PATH = '/usr/bin/stockfish'
CP_LOSS_THRESHOLD = 150
STOCKFISH_TIME_LIMIT = 0.05 # Reduced for speed
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_FILE = os.path.join(OUTPUT_DIR, 'inputs.npz')
TARGETS_FILE = os.path.join(OUTPUT_DIR, 'targets.npz')
EVALS_FILE = os.path.join(OUTPUT_DIR, 'evals.npz')
PROCESSED_LOG = os.path.join(OUTPUT_DIR, 'processed_urls.json')

# Global variable for worker process
engine = None

def worker_init():
    """Initialize the Stockfish engine for the worker process."""
    global engine
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"Error initializing engine in worker: {e}")

def get_archives(username):
    """Fetch list of monthly archive URLs from Chess.com API."""
    url = f"https://api.chess.com/pub/player/{username}/games/archives"
    headers = {'User-Agent': f'ChessDataMiner/1.0 (username: {username})'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get('archives', [])
    except requests.RequestException as e:
        print(f"Error fetching archives: {e}")
        return []

def process_archive_download(url):
    """Download PGN data for a specific monthly archive."""
    if not url.endswith('/pgn'):
        url += '/pgn'
        
    headers = {'User-Agent': f'ChessDataMiner/1.0 (username: {USERNAME})'}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching archive {url}: {e}")
        return None

def board_to_tensor(board, move_number=1):
    """Convert board to 19x8x8 float32 array.
    Planes 0-5:   White pieces (P, N, B, R, Q, K)
    Planes 6-11:  Black pieces (P, N, B, R, Q, K)
    Plane 12:     Turn (all 1s if White to move, all 0s if Black)
    Plane 13:     White kingside castling right
    Plane 14:     White queenside castling right
    Plane 15:     Black kingside castling right
    Plane 16:     Black queenside castling right
    Plane 17:     En passant target square
    Plane 18:     Temporal encoding: min(fullmove_number / 60.0, 1.0)
    """
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    # Piece planes (0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            layer = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                layer += 6
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            tensor[layer, rank, file] = 1.0

    # Turn plane (12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling rights (13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # En passant plane (17)
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        tensor[17, ep_rank, ep_file] = 1.0

    # Temporal plane (18): normalized full move number
    tensor[18, :, :] = min(move_number / 60.0, 1.0)

    return tensor

def encode_move(move):
    """Convert move to integer index: from_sq * 64 + to_sq."""
    return move.from_square * 64 + move.to_square

def mirror_board_and_move(board_tensor, move_from, move_to, target_eval):
    """Mirror the board tensor and move horizontally.
    Works with all 19 planes: piece planes flip positions,
    scalar planes (turn/castling/temporal) are uniform so flip is no-op,
    en passant square flips correctly.
    Eval is unchanged by horizontal mirror.
    """
    new_tensor = np.flip(board_tensor, axis=2).copy()
    def flip_h(sq):
        r = chess.square_rank(sq)
        f = chess.square_file(sq)
        return chess.square(7 - f, r)
    new_from = flip_h(move_from)
    new_to = flip_h(move_to)
    return new_tensor, encode_move(chess.Move(new_from, new_to)), target_eval

def color_reverse_tensor(board_tensor, move_from, move_to, target_eval):
    """180-degree rotation: play the same position as the other color.
    1. Swap piece planes: White (0-5) <-> Black (6-11)
    2. Flip board 180° (both axes) so pawns face the right direction
    3. Invert turn plane (12)
    4. Swap castling planes: White K/Q (13,14) <-> Black K/Q (15,16)
    5. En passant (17) gets flipped by the 180° rotation automatically
    6. Temporal plane (18) is uniform, unaffected by rotation
    7. Move squares: sq ^ 63 (flip both rank and file)
    8. Eval is NEGATED: we swapped who is playing
    """
    new_tensor = np.zeros_like(board_tensor)

    # Swap piece planes and flip 180°
    new_tensor[0:6]  = np.flip(board_tensor[6:12], axis=(1, 2))   # Black pieces -> White slots
    new_tensor[6:12] = np.flip(board_tensor[0:6],  axis=(1, 2))   # White pieces -> Black slots

    # Invert turn plane
    new_tensor[12] = 1.0 - board_tensor[12]

    # Swap castling planes (no flip needed, they're uniform)
    new_tensor[13] = board_tensor[15]  # Black K-side -> White K-side
    new_tensor[14] = board_tensor[16]  # Black Q-side -> White Q-side
    new_tensor[15] = board_tensor[13]  # White K-side -> Black K-side
    new_tensor[16] = board_tensor[14]  # White Q-side -> Black Q-side

    # En passant: flip 180°
    new_tensor[17] = np.flip(board_tensor[17], axis=(0, 1)).copy()

    # Temporal plane: uniform value, copy straight through
    new_tensor[18] = board_tensor[18]

    # Move: 180° rotation = sq ^ 63
    new_from = move_from ^ 63
    new_to = move_to ^ 63
    return new_tensor, encode_move(chess.Move(new_from, new_to)), -target_eval

def process_single_game(pgn_text):
    """
    Worker function to process a single game PGN.
    Returns a list of (tensor, move_idx, eval) tuples.
    Each valid position generates 4 samples:
      1. Original
      2. Horizontal mirror
      3. Color-reversed (180° rotation) — eval negated
      4. Color-reversed + horizontal mirror — eval negated
    """
    global engine
    if engine is None:
        return []
    
    results = []
    pgn_io = io.StringIO(pgn_text)
    
    try:
        game = chess.pgn.read_game(pgn_io)
    except Exception:
        return results
        
    if game is None:
        return results

    if game.headers.get("Variant", "Standard") != "Standard":
        return results

    board = game.board()
    
    white = game.headers.get("White", "").lower()
    username_lower = USERNAME.lower()
    if white == username_lower:
        our_color = chess.WHITE
    elif game.headers.get("Black", "").lower() == username_lower:
        our_color = chess.BLACK
    else:
        return results # Not our game

    node = game
    while not node.is_end():
        next_node = node.next()
        if not next_node:
            break
            
        move = next_node.move
        
        if board.turn == our_color:
            try:
                # Pre-move analysis (absolute eval from White's perspective)
                info_pre = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
                score_pre = info_pre["score"].white().score(mate_score=10000)
                
                if score_pre is not None:
                     board.push(move)
                     info_post = engine.analyse(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
                     score_post = info_post["score"].white().score(mate_score=10000)
                     board.pop()
                     
                     if score_post is not None:
                        loss = (score_pre - score_post) if our_color == chess.WHITE else (score_post - score_pre)
                        
                        if loss <= CP_LOSS_THRESHOLD:
                            # Convert absolute eval to relative (side-to-move advantage)
                            relative_cp = score_pre if board.turn == chess.WHITE else -score_pre
                            # Normalize to [-1, 1] using tanh
                            target_eval = math.tanh(relative_cp / 400.0)

                            tensor = board_to_tensor(board, board.fullmove_number)
                            move_idx = encode_move(move)
                            from_sq = move.from_square
                            to_sq = move.to_square

                            # 1. Original
                            results.append((tensor, move_idx, target_eval))
                            
                            # 2. Horizontal mirror (eval unchanged)
                            aug_tensor, aug_move_idx, aug_eval = mirror_board_and_move(tensor, from_sq, to_sq, target_eval)
                            results.append((aug_tensor, aug_move_idx, aug_eval))

                            # 3. Color-reversed (eval negated)
                            cr_tensor, cr_move_idx, cr_eval = color_reverse_tensor(tensor, from_sq, to_sq, target_eval)
                            results.append((cr_tensor, cr_move_idx, cr_eval))

                            # 4. Color-reversed + horizontal mirror (eval negated)
                            crm_tensor, crm_move_idx, crm_eval = mirror_board_and_move(cr_tensor, from_sq ^ 63, to_sq ^ 63, cr_eval)
                            results.append((crm_tensor, crm_move_idx, crm_eval))
            except Exception:
                pass # Skip problematic positions

        board.push(move)
        node = next_node
        
    return results

def split_pgn_text(pgn_full_text):
    """
    Splits a large string of multiple PGN games into a list of single PGN strings.
    This is a heuristic split on '[Event "'.
    """
    # Simply using a regex or split might be fragile if [Event " appears in comments.
    # But for Chess.com PGN downloads, it's usually clean.
    # A safer way might be to iterate line by line, but splitting on '\n[Event "' is reasonably safe.
    # We add the newline back.
    
    games = []
    current_game = []
    
    for line in pgn_full_text.splitlines():
        if line.startswith('[Event "'):
            if current_game:
                games.append("\n".join(current_game))
                current_game = []
        current_game.append(line)
        
    if current_game:
        games.append("\n".join(current_game))
        
    return games

def main():
    # 0. Fresh start — Phase 8 requires re-mine (19-plane tensor + evals)
    all_inputs = []
    all_targets = []
    all_evals = []
    processed_urls = set()

    if not os.path.exists(STOCKFISH_PATH):
        print(f"Stockfish not found at {STOCKFISH_PATH}. Please check path.")
        return

    # 1. Get Archives
    archives = get_archives(USERNAME)
    print(f"Found {len(archives)} monthly archives.")
    
    # Use 90% of cores or max-1
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    print(f"Starting stats mining with {num_processes} worker processes...")

    try:
        with multiprocessing.Pool(processes=num_processes, initializer=worker_init) as pool:
            
            for url in tqdm(archives, desc="Archives"):
                if url in processed_urls:
                    continue

                pgn_text = process_archive_download(url)
                if not pgn_text:
                    continue
                
                # Split games
                game_texts = split_pgn_text(pgn_text)
                if not game_texts:
                    continue
                    
                # Process in parallel
                results_nested = list(tqdm(pool.imap(process_single_game, game_texts, chunksize=5), 
                                           total=len(game_texts), 
                                           desc="Processing Games", 
                                           leave=False))
                
                # Flatten results (now 3-tuples: tensor, move_idx, eval)
                for game_results in results_nested:
                    for tensor, move_idx, target_eval in game_results:
                        all_inputs.append(tensor)
                        all_targets.append(move_idx)
                        all_evals.append(target_eval)
                
                # Incremental Save
                processed_urls.add(url)
                
                temp_inputs = os.path.join(OUTPUT_DIR, 'inputs_temp.npz')
                temp_targets = os.path.join(OUTPUT_DIR, 'targets_temp.npz')
                temp_evals = os.path.join(OUTPUT_DIR, 'evals_temp.npz')
                
                np.savez_compressed(temp_inputs, np.array(all_inputs, dtype=np.float32))
                np.savez_compressed(temp_targets, np.array(all_targets, dtype=np.int16))
                np.savez_compressed(temp_evals, np.array(all_evals, dtype=np.float32))
                
                os.replace(temp_inputs, INPUTS_FILE)
                os.replace(temp_targets, TARGETS_FILE)
                os.replace(temp_evals, EVALS_FILE)
                
                with open(PROCESSED_LOG, 'w') as f:
                    json.dump(list(processed_urls), f)
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        print(f"Final dataset size: {len(all_inputs)} samples.")
        if len(all_inputs) > 0:
            np.savez_compressed(INPUTS_FILE, np.array(all_inputs, dtype=np.float32))
            np.savez_compressed(TARGETS_FILE, np.array(all_targets, dtype=np.int16))
            np.savez_compressed(EVALS_FILE, np.array(all_evals, dtype=np.float32))
            with open(PROCESSED_LOG, 'w') as f:
                json.dump(list(processed_urls), f)
            print(f"Saved to {INPUTS_FILE}, {TARGETS_FILE}, and {EVALS_FILE}")

if __name__ == "__main__":
    main()
