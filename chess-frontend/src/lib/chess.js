import { Chess } from 'chess.js';

export const FILES = 'abcdefgh';
export const NAMES = { p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' };
const VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };

export const pieceSrc = (type, color) => `/pieces/${color}${type.toUpperCase()}.svg`;
export const formatScore = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
export const newGame = () => new Chess();

export function copyGame(game) {
  const copy = newGame();
  for (const move of game.history({ verbose: true })) copy.move({ from: move.from, to: move.to, promotion: move.promotion });
  return copy;
}

/** Position after the first `count` moves of a verbose history. */
export function positionAt(history, count) {
  const game = newGame();
  for (const move of history.slice(0, count)) game.move({ from: move.from, to: move.to, promotion: move.promotion });
  return game;
}

export function outcome(game) {
  if (!game.isGameOver()) return null;
  if (game.isCheckmate()) return { winner: game.turn() === 'w' ? 'black' : 'white', reason: 'Checkmate' };
  const reason = game.isStalemate() ? 'Stalemate' : game.isThreefoldRepetition() ? 'Threefold repetition' : game.isInsufficientMaterial() ? 'Insufficient material' : 'Draw';
  return { winner: 'draw', reason };
}

export function parseUci(uci) {
  return { from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] };
}

/** Standard notation for a UCI move in a position, or the UCI text if it does not apply. */
export function sanFor(fen, uci) {
  try { return new Chess(fen).move(parseUci(uci)).san; } catch { return uci; }
}

/** Grid cell (0-7 from the top-left of the screen) of a square for a given orientation. */
export function squareToCell(square, orientation) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return orientation === 'white' ? { col: file, row: 7 - rank } : { col: 7 - file, row: rank };
}

export function cellToSquare(col, row, orientation) {
  const file = orientation === 'white' ? col : 7 - col;
  const rank = orientation === 'white' ? 7 - row : row;
  return FILES[file] + (rank + 1);
}

const ROOK_CASTLE = {
  wk: ['h1', 'f1'], wq: ['a1', 'd1'], bk: ['h8', 'f8'], bq: ['a8', 'd8'],
};

/**
 * Piece lists for every position of a game, index 0 being the start. Each piece keeps the
 * same id for its whole life (through captures of others, castling, en passant and
 * promotion), so the board can animate real movement instead of redrawing pieces.
 */
export function buildSnapshots(history) {
  let id = 0;
  const back = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'];
  let pieces = [];
  for (let f = 0; f < 8; f++) {
    pieces.push({ id: id++, type: back[f], color: 'w', square: FILES[f] + '1' });
    pieces.push({ id: id++, type: 'p', color: 'w', square: FILES[f] + '2' });
    pieces.push({ id: id++, type: 'p', color: 'b', square: FILES[f] + '7' });
    pieces.push({ id: id++, type: back[f], color: 'b', square: FILES[f] + '8' });
  }
  const snapshots = [pieces];
  for (const move of history) {
    const capturedAt = move.flags.includes('e') ? move.to[0] + move.from[1] : move.to;
    const castle = move.flags.includes('k') ? ROOK_CASTLE[move.color + 'k'] : move.flags.includes('q') ? ROOK_CASTLE[move.color + 'q'] : null;
    pieces = pieces
      .filter(piece => !(move.captured && piece.square === capturedAt))
      .map(piece => {
        if (piece.square === move.from) return { ...piece, square: move.to, type: move.promotion ?? piece.type };
        if (castle && piece.square === castle[0]) return { ...piece, square: castle[1] };
        return piece;
      });
    snapshots.push(pieces);
  }
  return snapshots;
}

/**
 * The engine streams two kinds of progress. A completed-depth report ("Completed depth 3...")
 * carries the three best root moves so far. A probing report carries the one move being searched.
 */
export function classifyProgress(update) {
  if (!update?.preview_moves?.length) return null;
  return /^Completed/.test(update.message || '') ? 'ranked' : 'probing';
}

/** Pieces captured by one side, and the material lead (in pawns) from White's point of view. */
export function captureSummary(history) {
  const byWhite = [];
  const byBlack = [];
  let lead = 0;
  for (const move of history) {
    if (!move.captured) continue;
    (move.color === 'w' ? byWhite : byBlack).push(move.captured);
    lead += (move.color === 'w' ? 1 : -1) * VALUES[move.captured];
  }
  const order = 'qrbnp';
  const sort = list => list.sort((a, b) => order.indexOf(a) - order.indexOf(b));
  return { byWhite: sort(byWhite), byBlack: sort(byBlack), lead };
}

/** Plain-language reading of a score (pawns, White's view) from the player's side of the board. */
export function verdict(score, userColor) {
  const size = Math.abs(score);
  if (size < 0.35) return 'Level position';
  const degree = size < 1 ? 'slightly better' : size < 2.5 ? 'better' : 'winning';
  return (score > 0) === (userColor === 'w') ? `You're ${degree}` : `ResNet is ${degree}`;
}

/** Piyush Singh's 12-move win from 4 February 2023, replayed on the start screen. */
export const DEMO_MOVES = 'e2e4 d7d5 e4d5 d8d5 b1c3 d5e5 g1e2 g8h6 d2d4 e5f5 c1f4 h6g4 h2h3 g4f6 f4c7 b8c6 d4d5 c6b4 e2d4 e7e5 f1b5 e8e7 d5d6'.split(' ');
