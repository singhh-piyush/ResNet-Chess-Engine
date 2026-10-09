import { AnimatePresence, motion } from 'motion/react';
import { NAMES, pieceSrc } from '../lib/chess';
import Count from './Count';
import { ARRIVAL } from './Flyer';
import Tip from './Tip';

const ORDER = ['q', 'r', 'b', 'n', 'p'];
const SLIDE = { duration: 0.55, ease: [0.22, 1, 0.36, 1] };

export default function PlayerStrip({ name, color, captured, lead, status, active, reduced, className = '' }) {
  const groups = ORDER.map(type => ({ type, count: captured.filter(item => item === type).length })).filter(group => group.count);
  const takenColor = color === 'w' ? 'b' : 'w';
  const side = color === 'w' ? 'White' : 'Black';
  return (
    <div className={`strip ${className}${active ? ' active' : ''}`}>
      <strong className="name">{name}</strong>
      <span className="side-name">{side}</span>
      <AnimatePresence mode="wait" initial={false}>
        {status && <motion.span key={status} className="status" initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: 6 }} transition={{ duration: 0.2 }}>{status}</motion.span>}
      </AnimatePresence>
      <div className="tray" data-owner={color} aria-label={captured.length ? `${name} captured ${captured.map(type => NAMES[type]).join(', ')}` : undefined}>
        {/* A new piece stays hidden until the flying piece reaches its slot, then fades in under it;
            the pieces already there slide aside to make room rather than jumping. */}
        {groups.map(group => (
          <motion.div className="group" key={group.type} data-type={group.type} layout={reduced ? false : 'position'} transition={{ layout: SLIDE }}>
            {Array.from({ length: group.count }, (_, i) => (
              <motion.img
                key={i} className={takenColor} src={pieceSrc(group.type, takenColor)} alt=""
                initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                transition={reduced ? { duration: 0 } : { delay: ARRIVAL - 0.04, duration: 0.24, ease: 'easeOut' }}
              />
            ))}
          </motion.div>
        ))}
        <AnimatePresence initial={false}>
          {lead > 0 && (
            <Tip key="lead" content={`${lead} ${lead === 1 ? 'pawn' : 'pawns'} ahead in captured material`}>
              <motion.span
                className="lead" layout={reduced ? false : 'position'}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, transition: { duration: 0.3 } }}
                transition={{ layout: SLIDE, opacity: reduced ? { duration: 0 } : { delay: ARRIVAL, duration: 0.35 } }}
              >
                +<Count value={lead} duration={0.5} />
              </motion.span>
            </Tip>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
