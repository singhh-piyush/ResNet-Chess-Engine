import { memo, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, animate, motion, useMotionValue } from 'motion/react';
import { FILES, NAMES, cellToSquare, pieceSrc, squareToCell } from '../lib/chess';

const SPRING = { type: 'spring', stiffness: 230, damping: 27, mass: 0.9 };
const SETTLE = { type: 'spring', stiffness: 520, damping: 34 };
const pct = n => `${n * 100}%`;
const DRAG_THRESHOLD = 5;

/**
 * One piece. Its square is a pair of motion values (percent of its own size), so moves are
 * spring animations that never touch React state. `dx`/`dy` carry the live drag offset, and
 * a successful drop hands that offset over to the new square so the piece settles in place.
 */
const Piece = memo(function Piece({ piece, col, row, draggable, fallen, reduced, registryRef, settleRef }) {
  const x = useMotionValue(pct(col));
  const y = useMotionValue(pct(row));
  const dx = useMotionValue(0);
  const dy = useMotionValue(0);
  const z = useMotionValue(1);
  const root = useRef(null);
  const placed = useRef({ col, row });

  useEffect(() => {
    const registry = registryRef.current;
    registry.set(piece.id, { dx, dy, z, root });
    return () => registry.delete(piece.id);
  }, [piece.id, registryRef, dx, dy, z]);

  useLayoutEffect(() => {
    if (placed.current.col === col && placed.current.row === row) return;
    placed.current = { col, row };
    const handoff = settleRef.current;
    if (handoff && handoff.id === piece.id) {
      settleRef.current = null;
      x.set(pct(col)); y.set(pct(row));
      dx.set(handoff.dx); dy.set(handoff.dy);
      animate(dx, 0, SETTLE); animate(dy, 0, SETTLE);
      return;
    }
    if (reduced) { x.set(pct(col)); y.set(pct(row)); return; }
    z.set(6);
    animate(x, pct(col), SPRING);
    animate(y, pct(row), SPRING).then(() => z.set(1));
  }, [col, row, piece.id, reduced, settleRef, x, y, dx, dy, z]);

  return (
    <motion.div
      ref={root}
      className={`piece${draggable ? ' grab' : ''}`}
      style={{ x, y, zIndex: z }}
      initial={{ scale: 0.3, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0, delay: reduced ? 0 : 0.22 } }}
    >
      <motion.div
        className="piece-art" style={{ x: dx, y: dy, transformOrigin: '50% 88%' }}
        animate={{ rotate: fallen ? 82 : 0, opacity: fallen ? 0.7 : 1 }}
        transition={reduced ? { duration: 0 } : { type: 'spring', stiffness: 50, damping: 9, delay: fallen ? 0.5 : 0 }}
      >
        <motion.img key={piece.type} src={pieceSrc(piece.type, piece.color)} alt="" draggable={false} initial={{ scale: 0.7 }} animate={{ scale: 1 }} />
      </motion.div>
    </motion.div>
  );
});

function Ghost({ piece, from, to, weight, orientation, reduced }) {
  const a = squareToCell(from, orientation);
  const b = squareToCell(to, orientation);
  return (
    <motion.div
      className="piece ghost"
      initial={{ x: pct(a.col), y: pct(a.row), opacity: 0 }}
      animate={{ x: pct(b.col), y: pct(b.row), opacity: 0.12 + weight * 0.5 }}
      exit={{ opacity: 0, transition: { duration: reduced ? 0 : 0.7, ease: 'easeOut' } }}
      transition={{ x: { duration: reduced ? 0 : 1.2, ease: [0.22, 1, 0.36, 1] }, y: { duration: reduced ? 0 : 1.2, ease: [0.22, 1, 0.36, 1] }, opacity: { duration: 0.9, ease: 'easeInOut' } }}
    >
      <div className="piece-art"><img src={pieceSrc(piece.type, piece.color)} alt="" draggable={false} /></div>
    </motion.div>
  );
}

function Promotion({ promotion, color, orientation, onPick, onCancel }) {
  const { col, row } = squareToCell(promotion.to, orientation);
  const fromTop = row < 4;
  const order = fromTop ? ['q', 'r', 'b', 'n'] : ['n', 'b', 'r', 'q'];
  return (
    <div className="promo-scrim" onPointerDown={event => { event.stopPropagation(); onCancel(); }}>
      <motion.div
        className="promo" role="group" aria-label="Promote pawn"
        style={{ left: `${col * 12.5}%`, top: fromTop ? 0 : '50%' }}
        initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
        onPointerDown={event => event.stopPropagation()}
      >
        {order.map((type, i) => (
          <motion.button
            key={type} type="button" aria-label={`Promote to ${NAMES[type]}`}
            initial={{ opacity: 0, y: fromTop ? -10 : 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.04 }}
            onClick={() => onPick(type)}
          >
            <img src={pieceSrc(type, color)} alt="" draggable={false} />
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}

export default function Board({
  pieces, orientation, color, interactive, selected, targets, lastMove, checkSquare, showCoords,
  promotion, ghosts, fallenSquare, reduced, onSelect, onMove, onPromote, onCancelPromotion, children,
}) {
  const boardRef = useRef(null);
  const registry = useRef(new Map());
  const settle = useRef(null);
  const drag = useRef(null);
  const instructions = useId();
  const [cursor, setCursor] = useState(null);
  const [focused, setFocused] = useState(false);
  const [hover, setHover] = useState(null);

  const cursorSquare = cursor ?? (color === 'w' ? 'e2' : 'e7');
  const bySquare = useMemo(() => new Map(pieces.map(piece => [piece.square, piece])), [pieces]);
  const targetMap = useMemo(() => new Map(targets.map(target => [target.to, target])), [targets]);

  const squareAt = (clientX, clientY) => {
    const rect = boardRef.current.getBoundingClientRect();
    const col = Math.floor(((clientX - rect.left) / rect.width) * 8);
    const row = Math.floor(((clientY - rect.top) / rect.height) * 8);
    if (col < 0 || col > 7 || row < 0 || row > 7) return null;
    return cellToSquare(col, row, orientation);
  };

  const release = (id, snap) => {
    const entry = registry.current.get(id);
    if (!entry) return;
    entry.root.current?.classList.remove('lifted');
    if (snap) { animate(entry.dx, 0, SETTLE); animate(entry.dy, 0, SETTLE); }
    animate(entry.z, 1, { duration: 0.2 });
  };

  const onPointerDown = event => {
    if (promotion || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const square = squareAt(event.clientX, event.clientY);
    if (!square) return;
    setCursor(square);
    if (!interactive) return;
    const piece = bySquare.get(square);
    if (piece && piece.color === color) {
      onSelect(square);
      drag.current = { id: piece.id, from: square, sx: event.clientX, sy: event.clientY, active: false, wasSelected: selected === square };
      boardRef.current.setPointerCapture(event.pointerId);
    } else if (selected && targetMap.has(square)) {
      onMove(selected, square);
    } else {
      onSelect(null);
    }
  };

  const onPointerMove = event => {
    const d = drag.current;
    if (!d) return;
    const moved = Math.hypot(event.clientX - d.sx, event.clientY - d.sy);
    if (!d.active && moved < DRAG_THRESHOLD) return;
    const entry = registry.current.get(d.id);
    if (!entry) return;
    if (!d.active) {
      d.active = true;
      entry.root.current?.classList.add('lifted');
      entry.z.set(10);
    }
    entry.dx.set(event.clientX - d.sx);
    entry.dy.set(event.clientY - d.sy);
    const over = squareAt(event.clientX, event.clientY);
    setHover(prev => (prev === over ? prev : over));
  };

  const finishDrag = (event, cancelled) => {
    const d = drag.current;
    drag.current = null;
    setHover(null);
    if (boardRef.current.hasPointerCapture?.(event.pointerId)) boardRef.current.releasePointerCapture(event.pointerId);
    if (!d) return;
    if (!d.active) { if (d.wasSelected && !cancelled) onSelect(null); return; }
    const to = cancelled ? null : squareAt(event.clientX, event.clientY);
    const entry = registry.current.get(d.id);
    if (to && to !== d.from && targetMap.has(to) && entry) {
      const rect = boardRef.current.getBoundingClientRect();
      const size = rect.width / 8;
      const before = squareToCell(d.from, orientation);
      const after = squareToCell(to, orientation);
      settle.current = { id: d.id, dx: entry.dx.get() + (before.col - after.col) * size, dy: entry.dy.get() + (before.row - after.row) * size };
      entry.root.current?.classList.remove('lifted');
      if (onMove(d.from, to)) { animate(entry.z, 1, { duration: 0.3 }); return; }
      settle.current = null;
    }
    release(d.id, true);
  };

  const activate = square => {
    if (!interactive) return;
    const piece = bySquare.get(square);
    if (selected && targetMap.has(square)) onMove(selected, square);
    else if (piece && piece.color === color) onSelect(selected === square ? null : square);
    else onSelect(null);
  };

  const onKeyDown = event => {
    setFocused(true);
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); activate(cursorSquare); }
    else if (event.key === 'Escape') onSelect(null);
    else if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
      event.preventDefault();
      const direction = orientation === 'white' ? 1 : -1;
      const file = cursorSquare.charCodeAt(0) + direction * (event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0);
      const rank = Number(cursorSquare[1]) + direction * (event.key === 'ArrowUp' ? 1 : event.key === 'ArrowDown' ? -1 : 0);
      if (file >= 97 && file <= 104 && rank >= 1 && rank <= 8) setCursor(String.fromCharCode(file) + rank);
    }
  };

  const under = bySquare.get(cursorSquare);
  const label = `Chessboard. Cursor on ${cursorSquare}${under ? `, ${under.color === 'w' ? 'white' : 'black'} ${NAMES[under.type]}` : ', empty'}${selected ? `. ${selected} selected` : ''}.`;
  const origin = selected ? { file: selected.charCodeAt(0) - 97, rank: Number(selected[1]) } : null;

  return (
    <div
      ref={boardRef} className="board" role="group" tabIndex={0} aria-label={label} aria-describedby={instructions}
      onPointerDown={onPointerDown} onPointerMove={onPointerMove}
      onPointerUp={event => finishDrag(event, false)} onPointerCancel={event => finishDrag(event, true)}
      onKeyDown={onKeyDown}
      onFocus={event => { if (event.target.matches(':focus-visible')) setFocused(true); }}
      onBlur={() => setFocused(false)}
    >
      <div className="squares">
        {Array.from({ length: 64 }, (_, i) => {
          const col = i % 8, row = Math.floor(i / 8);
          const square = cellToSquare(col, row, orientation);
          const file = FILES.indexOf(square[0]), rank = Number(square[1]);
          const target = targetMap.get(square);
          const cls = ['sq', (file + rank) % 2 === 1 ? 'dark' : 'light'];
          if (lastMove && (square === lastMove.from || square === lastMove.to)) cls.push('last');
          if (square === selected) cls.push('sel');
          if (target) cls.push(target.capture ? 'cap' : 'tgt');
          if (target && hover === square) cls.push('hover');
          if (square === checkSquare) cls.push('chk');
          if (focused && square === cursorSquare) cls.push('cur');
          const distance = origin ? Math.max(Math.abs(file - origin.file), Math.abs(rank - origin.rank)) : 0;
          return (
            <div key={square} className={cls.join(' ')} data-square={square} style={{ '--d': `${distance * 24}ms` }}>
              {showCoords && col === 0 && <span className="coord rank">{rank}</span>}
              {showCoords && row === 7 && <span className="coord file">{square[0]}</span>}
            </div>
          );
        })}
      </div>
      <div className="pieces">
        <AnimatePresence initial={false}>
          {pieces.map(piece => {
            const { col, row } = squareToCell(piece.square, orientation);
            return <Piece key={piece.id} piece={piece} col={col} row={row} draggable={interactive && piece.color === color} fallen={piece.square === fallenSquare} reduced={reduced} registryRef={registry} settleRef={settle} />;
          })}
        </AnimatePresence>
        <AnimatePresence>
          {ghosts.map(ghost => {
            const piece = bySquare.get(ghost.from);
            return piece && <Ghost key={ghost.from + ghost.to} piece={piece} from={ghost.from} to={ghost.to} weight={ghost.weight} orientation={orientation} reduced={reduced} />;
          })}
        </AnimatePresence>
      </div>
      {children}
      <AnimatePresence>
        {promotion && <Promotion key="promotion" promotion={promotion} color={color} orientation={orientation} onPick={onPromote} onCancel={onCancelPromotion} />}
      </AnimatePresence>
      <p id={instructions} className="sr-only">Use arrow keys to move the cursor. Enter or Space selects a piece and then its destination. Escape clears the selection.</p>
    </div>
  );
}
