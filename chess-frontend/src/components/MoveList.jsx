import { useEffect, useRef } from 'react';
import { motion } from 'motion/react';
import { CaretDoubleLeft, CaretDoubleRight, CaretLeft, CaretRight } from '@phosphor-icons/react';
import Tip from './Tip';

export default function MoveList({ history, view, onView, reduced, botColor }) {
  const total = history.length;
  const current = view ?? total;
  const list = useRef(null);
  const go = index => onView(Math.max(0, Math.min(total, index)));

  useEffect(() => {
    const box = list.current;
    const active = box?.querySelector('[aria-current="true"]');
    if (!box || !active) { if (box) box.scrollTo({ top: 0, left: 0 }); return; }
    box.scrollTo({
      top: active.offsetTop - box.clientHeight / 2 + active.offsetHeight / 2,
      left: active.offsetLeft - box.clientWidth / 2 + active.offsetWidth / 2,
      behavior: reduced ? 'auto' : 'smooth',
    });
  }, [total, current, reduced]);

  const pairs = Array.from({ length: Math.ceil(total / 2) }, (_, i) => ({ number: i + 1, white: history[i * 2], black: history[i * 2 + 1], whitePly: i * 2 + 1 }));
  const ply = (move, index) => move && (
    <motion.button
      type="button" className={`ply${move.color === botColor ? ' bot' : ''}`} aria-current={current === index}
      initial={reduced ? false : { opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      onClick={() => onView(index >= total ? null : index)}
    >{move.san}</motion.button>
  );
  const reviewing = current < total;

  return (
    <section className="moves" aria-label="Moves">
      <div className="heading">
        <span className="label">Moves</span>
        <span className={`state${reviewing ? ' on' : ''}`}>{reviewing ? `Reviewing ${current} of ${total}` : total ? `Move ${Math.floor(total / 2) + 1}` : ''}</span>
      </div>
      <ol className="list" ref={list}>
        {!pairs.length && <li className="empty"><span className="wide">Your moves and ResNet's appear here. Click any move to look back at it.</span><span className="narrow">Moves will appear here</span></li>}
        {pairs.map(pair => (
          <li key={pair.number}>
            <span className="no">{pair.number}</span>
            {ply(pair.white, pair.whitePly)}
            {ply(pair.black, pair.whitePly + 1)}
          </li>
        ))}
      </ol>
      <div className="review" role="group" aria-label="Review">
        <Tip content="Start position"><button type="button" className="tool" aria-label="First position" disabled={current === 0} onClick={() => go(0)}><CaretDoubleLeft size={16} /></button></Tip>
        <Tip content="Previous move"><button type="button" className="tool" aria-label="Previous move" disabled={current === 0} onClick={() => go(current - 1)}><CaretLeft size={16} /></button></Tip>
        <Tip content="Next move"><button type="button" className="tool" aria-label="Next move" disabled={current === total} onClick={() => go(current + 1)}><CaretRight size={16} /></button></Tip>
        <Tip content="Back to the game"><button type="button" className="tool" aria-label="Latest position" disabled={current === total} onClick={() => onView(null)}><CaretDoubleRight size={16} /></button></Tip>
      </div>
    </section>
  );
}
