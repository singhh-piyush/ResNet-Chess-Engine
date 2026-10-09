import { useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { ArrowCounterClockwise, ArrowsDownUp, Flag, Handshake } from '@phosphor-icons/react';
import Tip from './Tip';

const swap = {
  initial: { opacity: 0, y: 10, filter: 'blur(4px)' },
  animate: { opacity: 1, y: 0, filter: 'blur(0px)' },
  exit: { opacity: 0, y: -10, filter: 'blur(4px)' },
  transition: { duration: 0.22, ease: [0.16, 1, 0.3, 1] },
};

/** Game actions along the bottom. New game (mid-game) and resign confirm in place. */
export default function Dock({ phase, canDraw, drawPending, onNew, onFlip, onDraw, onResign, reduced }) {
  const [confirm, setConfirm] = useState(null);
  const asking = phase === 'playing' ? confirm : null;
  const motionProps = reduced ? {} : swap;

  return (
    <motion.div className="dock glass" role="toolbar" aria-label="Game actions" layout={!reduced} transition={{ type: 'spring', stiffness: 420, damping: 36 }} onKeyDown={event => { if (event.key === 'Escape') setConfirm(null); }}>
      <AnimatePresence mode="popLayout" initial={false}>
        {asking ? (
          <motion.div key="confirm" className="dock-row" {...motionProps}>
            <span className="ask">{asking === 'new' ? 'Leave this game and start over?' : 'Resign this game?'}</span>
            <button type="button" className="btn ghost" autoFocus onClick={() => setConfirm(null)}>Keep playing</button>
            <button type="button" className="btn danger" onClick={() => { setConfirm(null); (asking === 'new' ? onNew : onResign)(); }}>{asking === 'new' ? 'New game' : 'Resign'}</button>
          </motion.div>
        ) : (
          <motion.div key="actions" className="dock-row" {...motionProps}>
            <Tip content="Start over and choose a side">
              <button type="button" className={`act${phase === 'finished' ? ' primary' : ''}`} onClick={() => (phase === 'playing' ? setConfirm('new') : onNew())}><ArrowCounterClockwise size={18} /><span>New game</span></button>
            </Tip>
            <Tip content="Turn the board around">
              <button type="button" className="act" onClick={onFlip}><ArrowsDownUp size={18} /><span>Flip</span></button>
            </Tip>
            <Tip content="Ask ResNet to agree to a draw. It accepts only when it doesn’t think it is ahead.">
              <button type="button" className="act" disabled={!canDraw} onClick={onDraw}><Handshake size={18} /><span>{drawPending ? 'Offering' : 'Offer draw'}</span></button>
            </Tip>
            <Tip content="Concede the game. You can still review every move.">
              <button type="button" className="act" disabled={phase !== 'playing'} onClick={() => setConfirm('resign')}><Flag size={18} /><span>Resign</span></button>
            </Tip>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
