"""Versioned absolute-color features and promotion-safe move vocabulary."""
import chess
import numpy as np

PROMOTIONS = tuple(chess.Move(chess.square(f, rank), chess.square(t, dest), promotion=p)
    for rank, dest in ((6, 7), (1, 0)) for f in range(8)
    for t in range(max(0, f-1), min(8, f+2))
    for p in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT))
PROMOTION_INDEX = {move: 4096+i for i, move in enumerate(PROMOTIONS)}
POLICY_SIZE = 4096 + len(PROMOTIONS)


def encode_move(move):
    return PROMOTION_INDEX[move] if move.promotion else move.from_square*64+move.to_square


def decode_move(index):
    if not 0 <= index < POLICY_SIZE:
        raise ValueError('Move index outside vocabulary')
    return PROMOTIONS[index-4096] if index >= 4096 else chess.Move(index//64, index%64)


def board_to_tensor(board):
    tensor = np.zeros((19, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        layer = piece.piece_type-1 + (0 if piece.color else 6)
        tensor[layer, chess.square_rank(square), chess.square_file(square)] = 1
    tensor[12] = board.turn
    for layer, color, kingside in ((13,True,True),(14,True,False),(15,False,True),(16,False,False)):
        tensor[layer] = (board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color))
    if board.ep_square is not None:
        tensor[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1
    tensor[18] = min(board.fullmove_number/60, 1)
    return tensor
