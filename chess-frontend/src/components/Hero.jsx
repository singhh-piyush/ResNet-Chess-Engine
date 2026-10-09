import { motion } from 'motion/react';
import { pieceSrc } from '../lib/chess';
import Count from './Count';
import Tip from './Tip';

const EASE = [0.16, 1, 0.3, 1];
const rise = { hidden: { y: '105%' }, show: { y: 0, transition: { duration: 1, ease: EASE } } };
const fade = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: EASE } } };

const FACTS = [
  { value: 3451, label: 'games learned', tip: 'My games that it studied, move by move.' },
  { value: 15, label: 'residual blocks', tip: 'Layers of the network that read each position. More blocks let it pick up subtler patterns.' },
  { value: 9.5, decimals: 1, label: 'seconds per move', tip: 'Its thinking time each turn, looking up to four moves ahead.' },
];

export default function Hero({ onPlay, reduced }) {
  return (
    <motion.section
      className="hero" aria-label="New game"
      initial={reduced ? false : 'hidden'} animate="show" exit={reduced ? undefined : { opacity: 0, x: -60, transition: { duration: 0.45, ease: [0.4, 0, 1, 1] } }}
      variants={{ show: { transition: { staggerChildren: 0.09, delayChildren: 0.15 } } }}
    >
      <h1>
        <span className="line"><motion.span variants={rise}>ResNet</motion.span></span>
        <span className="line soft"><motion.span variants={rise}>Chess Engine</motion.span></span>
      </h1>
      <motion.p className="tagline" variants={fade}>
        Trained on my games to play like me. Pick a side and try it.
      </motion.p>
      <motion.dl className="facts" variants={fade}>
        {FACTS.map(fact => (
          <Tip key={fact.label} content={fact.tip}>
            <div tabIndex={0}>
              <dt>{fact.label}</dt>
              <dd><Count value={fact.value} from={reduced ? fact.value : 0} decimals={fact.decimals || 0} duration={1.6} /></dd>
            </div>
          </Tip>
        ))}
      </motion.dl>
      <motion.div className="sides" variants={fade}>
        <button type="button" className="choose light" onClick={() => onPlay('white')}>
          <img src={pieceSrc('k', 'w')} alt="" />
          <span><strong>Play White</strong><small>You move first</small></span>
        </button>
        <button type="button" className="choose dark" onClick={() => onPlay('black')}>
          <img src={pieceSrc('k', 'b')} alt="" />
          <span><strong>Play Black</strong><small>ResNet opens</small></span>
        </button>
      </motion.div>
    </motion.section>
  );
}
