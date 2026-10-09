import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, animate, motion, useMotionValue, useReducedMotion } from 'motion/react';
import { Info } from '@phosphor-icons/react';
import { formatScore, verdict } from '../lib/chess';
import Count from './Count';
import Tip from './Tip';

export const MAX_DEPTH = 4;
const W = 200, H = 56, PAD = 6;
const toY = score => PAD + (0.5 - 0.5 * Math.tanh(score / 3)) * (H - PAD * 2);

const STATUS = {
  SELECTED: ['Played', 'The move ResNet chose.'],
  ANALYZED: ['Close', 'Within a pawn of the best line, so it stayed in the running.'],
  VETOED: ['Dropped', 'More than a pawn worse than the best line once ResNet looked ahead.'],
};

/** Smooth path through the points (Catmull-Rom converted to cubic curves). */
function smooth(points) {
  if (points.length < 2) return '';
  let d = `M${points[0][0]},${points[0][1]}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i - 1] || points[i], p1 = points[i], p2 = points[i + 1], p3 = points[i + 2] || p2;
    d += ` C${p1[0] + (p2[0] - p0[0]) / 6},${p1[1] + (p2[1] - p0[1]) / 6} ${p2[0] - (p3[0] - p1[0]) / 6},${p2[1] - (p3[1] - p1[1]) / 6} ${p2[0]},${p2[1]}`;
  }
  return d;
}

function Heading({ children, tip }) {
  return (
    <div className="heading">
      <Tip content={tip}><span className="label" tabIndex={0}>{children}<Info size={12} weight="bold" aria-hidden="true" /></span></Tip>
    </div>
  );
}

function Elapsed({ timer }) {
  const [now, setNow] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setNow(performance.now()), 100);
    return () => clearInterval(id);
  }, [timer]);
  return Math.max(0, Math.min(timer.duration, now - timer.start) / 1000).toFixed(1);
}

const toPoints = scores => scores.map((value, i) => [scores.length > 1 ? (i * W) / (scores.length - 1) : W, toY(value)]);

/**
 * The evaluation history. A new score never redraws the graph: the existing points ease left to
 * make room and the new one grows out of the previous end, so the line morphs in one motion.
 */
function Spark({ scores }) {
  const reduced = useReducedMotion();
  const line = useMotionValue('');
  const area = useMotionValue('');
  const left = useMotionValue('100%');
  const top = useMotionValue('50%');
  const shown = useRef([]);

  useEffect(() => {
    const draw = (points) => {
      shown.current = points;
      const d = smooth(points);
      line.set(d);
      area.set(d && `${d} L${W},${H} L0,${H} Z`);
      const last = points.at(-1);
      if (last) { left.set(`${(last[0] / W) * 100}%`); top.set(`${(last[1] / H) * 100}%`); }
    };
    const target = toPoints(scores);
    const from = shown.current;
    if (!target.length || !from.length || reduced) { draw(target); return undefined; }
    // Start from what is on screen now; extra points start where the line currently ends.
    const start = target.map((_, i) => from[Math.min(i, from.length - 1)]);
    const controls = animate(0, 1, {
      duration: 0.9, ease: [0.22, 1, 0.36, 1],
      onUpdate: t => draw(start.map(([x, y], i) => [x + (target[i][0] - x) * t, y + (target[i][1] - y) * t])),
    });
    return () => controls.stop();
  }, [scores, reduced, line, area, left, top]);

  return (
    <>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="spark-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="var(--accent)" stopOpacity="0.3" />
            <stop offset="1" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <line x1="0" x2={W} y1={H / 2} y2={H / 2} className="zero" />
        <motion.path d={area} fill="url(#spark-fill)" />
        <motion.path d={line} className="trace" />
      </svg>
      <motion.span className="dot" style={{ left, top }} initial={false} animate={{ opacity: scores.length ? 1 : 0 }} transition={{ duration: 0.4 }} />
    </>
  );
}

function Evaluation({ scores, userColor }) {
  const current = scores.at(-1) ?? 0;
  return (
    <section className="block evaluation" aria-label="Evaluation">
      <Heading tip="ResNet's estimate of the position after its last move, in pawns. Above zero favors White, below zero favors Black.">Evaluation</Heading>
      <div className="eval-row">
        <strong className={`big${scores.length ? '' : ' idle'}`}><Count value={current} decimals={2} signed duration={0.8} /></strong>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span key={scores.length ? verdict(current, userColor) : 'none'} className="verdict" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3, ease: 'easeInOut' }}>
            {scores.length ? verdict(current, userColor) : 'Updates after each ResNet move'}
          </motion.span>
        </AnimatePresence>
      </div>
      <Tip content="How the evaluation has moved over the game, one point per ResNet move. The dashed line is equal.">
        <div className="spark" role="img" aria-label={scores.length ? `Evaluation history, now ${current.toFixed(2)}` : 'No evaluations yet'}>
          <Spark scores={scores} />
        </div>
      </Tip>
    </section>
  );
}

function Search({ thinking, progress, search, timer, leader, stats, budget }) {
  const depth = thinking ? search.depth : stats?.depth ?? 0;
  const positions = thinking ? search.positions : stats?.positions ?? 0;
  const has = thinking || stats;
  const state = thinking
    ? progress?.phase === 'connecting' ? 'Connecting' : progress?.phase === 'waiting' ? 'Waiting for a free engine' : 'Thinking'
    : stats ? 'Last move' : 'Idle';
  const seconds = stats ? stats.elapsed_ms / 1000 : 0;
  return (
    <section className="block search" aria-label="Search">
      <div className="heading">
        <Tip content="What ResNet does on its turn: it reads the position, then looks ahead through the most promising moves until its time runs out.">
          <span className="label" tabIndex={0}>Search<Info size={12} weight="bold" aria-hidden="true" /></span>
        </Tip>
        <span className={`state${thinking ? ' on' : ''}`}>{state}</span>
      </div>
      <dl className="metrics">
        <Tip content={`How many moves ahead the search has looked, up to ${MAX_DEPTH}.`}>
          <div tabIndex={0}>
            <dt>Depth</dt>
            <dd>{has ? depth : '–'}<small>/{MAX_DEPTH}</small></dd>
            <span className="steps" aria-hidden="true">{Array.from({ length: MAX_DEPTH }, (_, i) => <i key={i} className={has && depth > i ? 'on' : ''} />)}</span>
          </div>
        </Tip>
        <Tip content="Positions checked so far. Each one is read by the network.">
          <div tabIndex={0}>
            <dt>Positions</dt>
            <dd>{has ? <Count value={positions} duration={0.4} /> : '–'}</dd>
          </div>
        </Tip>
        <Tip content={`Time used out of ${(budget / 1000).toFixed(1)} seconds.`}>
          <div tabIndex={0}>
            <dt>Time</dt>
            <dd>{thinking && timer ? <Elapsed timer={timer} /> : stats ? seconds.toFixed(1) : '–'}<small>s</small></dd>
            <span className="meter" aria-hidden="true">
              {thinking && timer
                ? <i key={timer.key} className="run" style={{ animationDuration: `${timer.duration}ms`, '--run': `${timer.duration}ms` }} />
                : <i style={{ transform: `scaleX(${stats ? Math.min(1, stats.elapsed_ms / budget) : 0})` }} />}
            </span>
          </div>
        </Tip>
      </dl>
      <AnimatePresence mode="wait" initial={false}>
        <motion.p
          key={thinking ? `lean-${leader?.san || ''}` : stats ? `played-${stats.lastSan}` : 'idle'} className="lean"
          initial={{ opacity: 0, y: 5 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -5 }} transition={{ duration: 0.22 }}
        >
          {thinking
            ? leader?.san ? <>Leaning toward <strong>{leader.san}</strong></> : 'Reading the position'
            : stats ? <>Played <strong>{stats.lastSan}</strong><span className="muted"> · {Math.round(stats.confidence * 100)}% instinct</span></> : 'Starts when it is ResNet’s turn'}
        </motion.p>
      </AnimatePresence>
    </section>
  );
}

function Candidates({ stats, thinking, botColor }) {
  const sign = botColor === 'w' ? 1 : -1;
  const rows = stats ? [...stats.candidates].sort((a, b) => (b.evaluation - a.evaluation) * sign || b.confidence - a.confidence) : [];
  const top = Math.max(...rows.map(row => row.confidence), 0.01);
  return (
    <section className={`block candidates${thinking && rows.length ? ' stale' : ''}`} aria-label="Candidate moves">
      <Heading tip="The moves ResNet weighed on its last turn, best first by evaluation.">Candidates</Heading>
      {rows.length ? (
        <>
          <div className="cand-head" aria-hidden="true">
            <span>Move</span>
            <Tip content="How likely I’d be to play this move, judged from the position alone before looking ahead."><span>Instinct</span></Tip>
            <Tip content="Score after looking ahead, in pawns. Above zero favors White."><span>Eval</span></Tip>
          </div>
          {/* Rows stay in place between turns: while the section is dimmed the text updates and each
              bar glides from its old length to the new one, so the list brightens instead of blinking. */}
          <ol className="cand-list">
            {rows.map((row, i) => {
              const [label, why] = STATUS[row.status] || STATUS.ANALYZED;
              return (
                <Tip key={i} content={<><strong>{label}.</strong> {why}</>}>
                  <li className={`cand ${row.status?.toLowerCase()}`} tabIndex={0}>
                    <span className="san"><i className="mark" aria-hidden="true" />{row.san || row.move}<span className="sr-only">, {label}</span></span>
                    <span className="inst">
                      <span className="bar"><motion.i initial={{ scaleX: 0 }} animate={{ scaleX: row.confidence / top }} transition={{ delay: Math.min(i, 8) * 0.03, duration: 0.8, ease: [0.22, 1, 0.36, 1] }} /></span>
                      <span className="pct">{(row.confidence * 100).toFixed(row.confidence < 0.1 ? 1 : 0)}%</span>
                    </span>
                    <span className="ev">{formatScore(row.evaluation)}</span>
                  </li>
                </Tip>
              );
            })}
          </ol>
        </>
      ) : <p className="empty">{thinking ? 'Appears once ResNet moves' : 'None yet. ResNet lists the moves it weighed after each turn.'}</p>}
    </section>
  );
}

/** Left panel: the position estimate, the live search, and the moves ResNet weighed. */
export default function Analysis({ scores, stats, thinking, progress, search, timer, leader, userColor, botColor, budget }) {
  return (
    <>
      <Evaluation scores={scores} userColor={userColor} />
      <Search thinking={thinking} progress={progress} search={search} timer={timer} leader={leader} stats={stats} budget={budget} />
      <Candidates stats={stats} thinking={thinking} botColor={botColor} />
    </>
  );
}
