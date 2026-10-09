import { motion } from 'motion/react';

export default function ResultCard({ result, side, moves, delay, onNew, onClose }) {
  const title = result.winner === 'draw' ? 'Draw' : result.winner === side ? 'You won' : 'ResNet won';
  const count = Math.ceil(moves / 2);
  return (
    <motion.div
      className={`result ${result.winner === 'draw' ? 'draw' : result.winner === side ? 'win' : 'loss'}`} role="dialog" aria-label={title}
      initial={{ opacity: 0, y: 28, scale: 0.94, filter: 'blur(8px)' }} animate={{ opacity: 1, y: 0, scale: 1, filter: 'blur(0px)' }} exit={{ opacity: 0, y: 12, filter: 'blur(4px)' }}
      transition={{ type: 'spring', stiffness: 160, damping: 22, delay }}
    >
      <h2>{title}</h2>
      <p>{result.reason}{count ? ` after ${count} ${count === 1 ? 'move' : 'moves'}` : ''}</p>
      <div className="result-actions">
        <button type="button" className="btn primary" onClick={onNew}>Play again</button>
        <button type="button" className="btn" onClick={onClose}>Review the game</button>
      </div>
    </motion.div>
  );
}
