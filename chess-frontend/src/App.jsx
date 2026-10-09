import { useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { ArrowClockwise, GearSix, Info, Moon, Sun, X } from '@phosphor-icons/react';
import { searchPosition, API_BASE } from './api';
import {
  DEMO_MOVES, buildSnapshots, captureSummary, classifyProgress, copyGame, newGame, outcome, parseUci,
  pieceSrc, positionAt, sanFor, squareToCell,
} from './lib/chess';
import Aura from './components/Aura';
import Board from './components/Board';
import FxLayer from './components/FxLayer';
import Flyer from './components/Flyer';
import PlayerStrip from './components/PlayerStrip';
import Analysis from './components/Analysis';
import MoveList from './components/MoveList';
import Dock from './components/Dock';
import Drawer from './components/Drawer';
import Hero from './components/Hero';
import ResultCard from './components/ResultCard';
import Tip from './components/Tip';

const BUDGET_MS = 3000;
const RANK_WEIGHT = [1, 0.5, 0.32];
const DROPPED_WEIGHT = 0.18;
const STORAGE_KEY = 'rce-settings';
const DEFAULTS = { fx: true, ghost: true, coords: true, theme: 'system' };
const EASE = [0.16, 1, 0.3, 1];
const FILL = '.btn, .act, .nav-btn, .choose, .tool';

const DEMO_HISTORY = (() => {
  const game = newGame();
  for (const uci of DEMO_MOVES) game.move(parseUci(uci));
  return game.history({ verbose: true });
})();
const DEMO_SNAPSHOTS = buildSnapshots(DEMO_HISTORY);

const loadSettings = () => {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}') }; } catch { return DEFAULTS; }
};

const lightQuery = '(prefers-color-scheme: light)';
const subscribeLight = callback => {
  const media = matchMedia(lightQuery);
  media.addEventListener('change', callback);
  return () => media.removeEventListener('change', callback);
};

/** Direction from the board's centre to a square, for the light behind the board. */
const angleTo = (square, orientation) => {
  const { col, row } = squareToCell(square, orientation);
  return Math.atan2(row + 0.5 - 4, col + 0.5 - 4);
};
const kingIn = (pieces, color) => pieces.find(piece => piece.type === 'k' && piece.color === color)?.square;

function Switch({ checked, onChange, label }) {
  return <button type="button" role="switch" aria-checked={checked} aria-label={label} className="switch" onClick={() => onChange(!checked)}><i /></button>;
}

function Setting({ title, children, control }) {
  return (
    <div className="setting">
      <div><strong>{title}</strong>{children && <p>{children}</p>}</div>
      {control}
    </div>
  );
}

export default function App() {
  const reduced = Boolean(useReducedMotion());
  const prefersLight = useSyncExternalStore(subscribeLight, () => matchMedia(lightQuery).matches, () => false);
  const [settings, setSettings] = useState(loadSettings);
  const theme = settings.theme === 'system' ? (prefersLight ? 'light' : 'dark') : settings.theme;

  const [game, setGame] = useState(newGame);
  const gameRef = useRef(game);
  const [side, setSide] = useState('white');
  const sideRef = useRef('white');
  const [phase, setPhase] = useState('setup');
  const [thinking, setThinking] = useState(false);
  const [progress, setProgress] = useState(null);
  const [search, setSearch] = useState({ depth: 0, positions: 0 });
  const [leader, setLeader] = useState(null);
  const [options, setOptions] = useState([]);
  const [timer, setTimer] = useState(null);
  const [stats, setStats] = useState(null);
  const [scores, setScores] = useState([]);
  const [flyers, setFlyers] = useState([]);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [result, setResult] = useState(null);
  const [card, setCard] = useState(false);
  const [panel, setPanel] = useState(null);
  const [selected, setSelected] = useState(null);
  const [promotion, setPromotion] = useState(null);
  const [flipped, setFlipped] = useState(false);
  const [view, setView] = useState(null);
  const [drawPending, setDrawPending] = useState(false);
  const [demoPly, setDemoPly] = useState(0);
  const [demoLeader, setDemoLeader] = useState(null);

  const controller = useRef(null);
  const generation = useRef(0);
  const fx = useRef(null);
  const aura = useRef(null);
  const timerKey = useRef(0);
  const latest = useRef({ depth: 0, positions: 0 });
  const optionsRef = useRef([]);
  const flush = useRef(0);
  const fxSeen = useRef(0);
  const orientationRef = useRef('white');

  useEffect(() => () => { controller.current?.abort(); clearTimeout(flush.current); }, []);

  // Buttons fill with colour from the point the pointer enters and drain toward where it leaves.
  useEffect(() => {
    const mark = event => {
      const button = event.target.closest?.(FILL);
      if (!button || button.contains(event.relatedTarget)) return;
      const rect = button.getBoundingClientRect();
      button.style.setProperty('--fx', `${event.clientX - rect.left}px`);
      button.style.setProperty('--fy', `${event.clientY - rect.top}px`);
    };
    document.addEventListener('pointerover', mark);
    document.addEventListener('pointerout', mark);
    return () => { document.removeEventListener('pointerover', mark); document.removeEventListener('pointerout', mark); };
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'light' ? '#E9ECF2' : '#07080B');
  }, [theme]);
  useEffect(() => { try { localStorage.setItem(STORAGE_KEY, JSON.stringify(settings)); } catch { /* Storage may be unavailable. */ } }, [settings]);
  const setting = key => value => setSettings(current => ({ ...current, [key]: value }));

  const setup = phase === 'setup';
  const history = useMemo(() => game.history({ verbose: true }), [game]);
  const snapshots = useMemo(() => buildSnapshots(history), [history]);
  const total = history.length;
  const viewing = view !== null && view < total;
  const shown = viewing ? view : total;
  const shownGame = useMemo(() => (viewing ? positionAt(history, view) : game), [viewing, history, view, game]);
  const orientation = setup ? 'white' : flipped ? (side === 'white' ? 'black' : 'white') : side;
  const userTurn = phase === 'playing' && game.turn() === side[0] && !thinking;
  const interactive = userTurn && !viewing && !promotion;
  const botColor = side === 'white' ? 'b' : 'w';
  const userColor = side[0];

  useEffect(() => { orientationRef.current = orientation; }, [orientation]);

  // What the board shows: the replayed demo on the start screen, otherwise the game (or a reviewed move).
  const demoLast = DEMO_HISTORY[demoPly - 1];
  const pieces = setup ? DEMO_SNAPSHOTS[demoPly] : snapshots[shown];
  const lastMove = setup ? demoLast ?? null : shown > 0 ? history[shown - 1] : null;
  const checkSquare = setup
    ? demoLast && /[+#]$/.test(demoLast.san) ? kingIn(pieces, demoLast.color === 'w' ? 'b' : 'w') : null
    : shownGame.isCheck() ? kingIn(snapshots[shown], shownGame.turn()) : null;
  const fallenSquare = setup
    ? demoPly === DEMO_HISTORY.length ? kingIn(pieces, 'b') : null
    : !viewing && result?.reason === 'Checkmate' ? kingIn(snapshots[total], game.turn()) : null;

  const targets = useMemo(() => {
    if (!selected || viewing) return [];
    const seen = new Map();
    for (const move of game.moves({ square: selected, verbose: true })) seen.set(move.to, { to: move.to, capture: Boolean(move.captured) });
    return [...seen.values()];
  }, [selected, viewing, game]);

  const { byWhite, byBlack, lead } = useMemo(() => captureSummary(history), [history]);

  // After a move: a light trail, then the landing ripple, capture burst and flight to the tray, check or mate shockwave.
  useEffect(() => {
    const last = history.at(-1);
    if (history.length < fxSeen.current) fxSeen.current = history.length;
    if (!last || history.length <= fxSeen.current) return undefined;
    const index = history.length;
    fx.current?.trail(last.from, last.to, last.color !== sideRef.current[0]);
    if (last.captured && !reduced) {
      const gone = snapshots[index - 1].find(piece => !snapshots[index].some(other => other.id === piece.id));
      const board = document.querySelector('.board');
      const tray = document.querySelector(`.tray[data-owner="${last.color}"]`);
      if (gone && board && tray) {
        const rect = board.getBoundingClientRect();
        const cell = squareToCell(gone.square, orientationRef.current);
        const size = rect.width / 8;
        const slot = tray.querySelector(`[data-type="${gone.type}"] img:last-child`) || tray;
        const x = rect.left + (cell.col + 0.5) * size, y = rect.top + (cell.row + 0.5) * size;
        setFlyers(list => [...list, { id: `${index}-${gone.id}`, src: pieceSrc(gone.type, gone.color), x, y, size: size * 0.92, slot }]);
      }
    }
    const timeout = setTimeout(() => {
      fxSeen.current = index;
      fx.current?.land(last.to);
      if (last.captured) fx.current?.burst(last.to, last.color === 'w' ? 'b' : 'w');
      const king = kingIn(snapshots[index], last.color === 'w' ? 'b' : 'w');
      if (last.san.endsWith('#')) fx.current?.shock(king, true);
      else if (last.san.endsWith('+')) fx.current?.shock(king, false);
    }, reduced ? 0 : 230);
    return () => clearTimeout(timeout);
  }, [history, snapshots, reduced]);

  // Start screen: replay one of Piyush's wins, with ResNet visibly thinking before each of Piyush's moves.
  useEffect(() => {
    if (phase !== 'setup' || reduced) return undefined;
    const layer = fx, glow = aura;
    const timeouts = [];
    const at = (ms, fn) => timeouts.push(setTimeout(fn, ms));
    const loop = () => {
      timeouts.splice(0);
      let clock = 1400;
      DEMO_HISTORY.forEach((move, ply) => {
        const think = move.color === 'w';
        if (think) {
          at(clock, () => glow.current?.start());
          at(clock + 450, () => { glow.current?.attend(angleTo(move.to, 'white'), 0.85); layer.current?.lead(move.to); setDemoLeader({ from: move.from, to: move.to }); });
          clock += 1800;
        } else clock += 900;
        at(clock, () => { setDemoLeader(null); if (think) glow.current?.decide(); glow.current?.stop(); layer.current?.stop(); setDemoPly(ply + 1); layer.current?.trail(move.from, move.to, think); });
        at(clock + 230, () => {
          layer.current?.land(move.to);
          if (move.captured) layer.current?.burst(move.to, move.color === 'w' ? 'b' : 'w');
          const king = kingIn(DEMO_SNAPSHOTS[ply + 1], move.color === 'w' ? 'b' : 'w');
          if (move.san.endsWith('#')) layer.current?.shock(king, true);
          else if (move.san.endsWith('+')) layer.current?.shock(king, false);
        });
        clock += 650;
      });
      at(clock + 3600, () => setDemoPly(0));
      at(clock + 5200, loop);
    };
    loop();
    return () => { timeouts.forEach(clearTimeout); layer.current?.clear(); glow.current?.clear(); setDemoLeader(null); };
  }, [phase, reduced]);

  const updateGame = next => { gameRef.current = next; setGame(next); };
  const resetThinking = () => {
    fx.current?.clear(); aura.current?.clear();
    optionsRef.current = []; setOptions([]);
    setLeader(null); setTimer(null); setSearch({ depth: 0, positions: 0 }); latest.current = { depth: 0, positions: 0 };
  };
  const finish = nextResult => {
    generation.current++; controller.current?.abort();
    setThinking(false); setPhase('finished'); setResult(nextResult); setCard(true); setSelected(null); setPromotion(null);
  };
  const cancelSearch = () => { generation.current++; controller.current?.abort(); controller.current = null; setThinking(false); };

  function handleProgress(update, fen) {
    if (update.phase === 'started') {
      aura.current?.start();
      setProgress(update);
      setTimer({ key: ++timerKey.current, start: performance.now(), duration: BUDGET_MS });
      return;
    }
    if (update.phase === 'waiting') { setProgress(update); return; }
    setProgress(current => (current?.phase === 'progress' ? current : { phase: 'progress' }));
    const kind = classifyProgress(update);
    latest.current = { depth: Math.max(latest.current.depth, update.depth ?? 0), positions: update.positions ?? latest.current.positions };
    if (!flush.current) flush.current = setTimeout(() => { flush.current = 0; setSearch({ ...latest.current }); }, 150);
    if (kind === 'probing') {
      aura.current?.attend(angleTo(parseUci(update.preview_moves[0]).to, orientationRef.current), 0.55);
    } else if (kind === 'ranked') {
      // Every move that has ranked near the top stays on the board until the decision;
      // the current favourite is brightest and the ones that slipped down just dim.
      const ranking = update.preview_moves;
      const next = optionsRef.current.map(option => ({ ...option, weight: ranking.includes(option.uci) ? RANK_WEIGHT[ranking.indexOf(option.uci)] : Math.min(option.weight, DROPPED_WEIGHT) }));
      ranking.forEach((uci, i) => { if (!next.some(option => option.uci === uci)) next.push({ uci, ...parseUci(uci), weight: RANK_WEIGHT[i] }); });
      optionsRef.current = next; setOptions(next);
      const glow = new Map();
      for (const option of next) glow.set(option.to, Math.max(glow.get(option.to) || 0, option.weight));
      for (const [square, weight] of glow) fx.current?.lead(square, weight * 0.8);
      const best = ranking[0];
      const { from, to } = parseUci(best);
      aura.current?.attend(angleTo(to, orientationRef.current), 0.9);
      setLeader(current => (current?.uci === best ? current : { uci: best, from, to, san: sanFor(fen, best) }));
    }
  }

  async function engineMove(position = gameRef.current) {
    if (controller.current || position.isGameOver()) return;
    const request = new AbortController();
    controller.current = request;
    const run = generation.current;
    const fen = position.fen();
    const began = performance.now();
    setThinking(true); setError(null); setNotice(null); setProgress({ phase: 'connecting' }); setSelected(null);
    resetThinking();
    const moves = position.history({ verbose: true }).map(move => move.from + move.to + (move.promotion || ''));
    try {
      const data = await searchPosition({ fen, moves }, { signal: request.signal, onProgress: update => { if (run === generation.current && !request.signal.aborted) handleProgress(update, fen); } });
      if (request.signal.aborted || run !== generation.current || gameRef.current.fen() !== fen) return;
      if (!data.move) { const end = outcome(position); if (end) finish(end); else throw new Error('ResNet returned no move. Try again.'); return; }
      // No pause at the decision: the other options fade as the chosen piece moves straight there.
      const chosen = parseUci(data.move);
      fx.current?.decide(data.move); aura.current?.decide();
      const next = copyGame(position);
      const move = next.move({ from: chosen.from, to: chosen.to, promotion: chosen.promotion });
      if (!move) throw new Error('ResNet returned an invalid move. Try again.');
      setStats({ ...data, elapsed_ms: data.elapsed_ms ?? performance.now() - began, lastSan: move.san });
      setScores(list => [...list, data.evaluation]);
      updateGame(next);
      const end = outcome(next); if (end) finish(end);
    } catch (failure) {
      if (!request.signal.aborted && run === generation.current) setError(failure.message || 'Could not reach ResNet. Try again.');
    } finally {
      if (controller.current === request) {
        controller.current = null; setThinking(false); setProgress(null);
        clearTimeout(flush.current); flush.current = 0;
        fx.current?.stop(); aura.current?.stop(); setLeader(null); setTimer(null);
        optionsRef.current = []; setOptions([]);
      }
    }
  }

  function playMove(from, to, promoted) {
    if (phase !== 'playing' || controller.current || gameRef.current.turn() !== sideRef.current[0]) return false;
    const legal = gameRef.current.moves({ square: from, verbose: true }).filter(move => move.to === to);
    if (!legal.length) return false;
    if (legal.some(move => move.promotion) && !promoted) { setPromotion({ from, to }); setSelected(null); return false; }
    try {
      const next = copyGame(gameRef.current);
      next.move({ from, to, promotion: promoted });
      updateGame(next); setSelected(null); setError(null); setNotice(null); setView(null);
      const end = outcome(next); if (end) finish(end); else engineMove(next);
      return true;
    } catch { return false; }
  }

  const resetGameState = () => {
    setStats(null); setScores([]); setResult(null); setCard(false); setError(null); setNotice(null); setProgress(null);
    setSelected(null); setPromotion(null); setFlipped(false); setDrawPending(false); setView(null); setFlyers([]);
    resetThinking();
  };

  function startGame(chosenSide) {
    cancelSearch();
    sideRef.current = chosenSide; setSide(chosenSide);
    const next = newGame(); updateGame(next);
    setPhase('playing'); setPanel(null); setDemoPly(0);
    resetGameState();
    if (chosenSide === 'black') engineMove(next);
  }

  function setupGame() {
    cancelSearch();
    setPhase('setup'); setDemoPly(0);
    resetGameState();
    updateGame(newGame());
  }

  async function offerDraw() {
    if (drawPending || thinking || phase !== 'playing') return;
    const run = generation.current;
    const offeredFen = game.fen();
    setDrawPending(true); setNotice(null);
    try {
      const response = await fetch(`${API_BASE}/offer_draw`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fen: game.fen(), user_side: side }) });
      if (!response.ok) throw new Error(response.status === 429 ? 'ResNet is busy. Offer again in a moment.' : 'The draw offer could not be sent. Try again.');
      const data = await response.json();
      if (run !== generation.current || gameRef.current.fen() !== offeredFen) return;
      if (data.accepted) finish({ winner: 'draw', reason: 'Draw agreed' });
      else setNotice('ResNet declined the draw. Play on.');
    } catch (failure) { if (run === generation.current) setNotice(failure.message); }
    finally { if (run === generation.current) setDrawPending(false); }
  }

  // Development only: jump straight to a position to exercise promotion, mate and captures.
  useEffect(() => {
    if (!import.meta.env.DEV) return undefined;
    window.__rce = {
      load(moves, chosenSide = 'white') {
        cancelSearch();
        const next = newGame();
        for (const uci of moves) next.move(parseUci(uci));
        sideRef.current = chosenSide; setSide(chosenSide); updateGame(next);
        setPhase('playing'); resetGameState();
      },
    };
    return () => { delete window.__rce; };
  });

  const botStatus = thinking ? (progress?.phase === 'connecting' ? 'Connecting' : progress?.phase === 'waiting' ? 'Waiting' : 'Thinking') : error ? 'Paused' : '';
  const userStatus = phase === 'finished' ? '' : userTurn ? (game.isCheck() ? 'Check' : 'Your move') : '';
  const announcement = phase === 'finished' ? result?.reason : thinking ? botStatus : userStatus || 'Waiting';
  const ghosts = !settings.ghost ? [] : setup ? (demoLeader ? [{ ...demoLeader, weight: 1 }] : []) : thinking && !reduced ? options : [];
  const showCard = phase === 'finished' && result && card && !viewing;
  const leadFor = color => (color === 'w' ? Math.max(lead, 0) : Math.max(-lead, 0));
  const blocked = panel ? '' : undefined;
  // Game chrome arrives after the board has started gliding into place.
  const enter = (delay, from) => (reduced ? {} : {
    initial: { opacity: 0, ...from },
    animate: { opacity: 1, x: 0, y: 0 },
    transition: { duration: 0.8, ease: EASE, delay: delay + 0.25 },
  });

  return (
    <div className={`app ${setup ? 'is-setup' : 'is-game'}`}>
      <header className="top" inert={blocked}>
        <AnimatePresence>
          {!setup && <motion.div key="brand" className="brand" exit={{ opacity: 0 }} {...enter(0.1, { y: -8 })}>ResNet<span>Chess Engine</span></motion.div>}
        </AnimatePresence>
        <nav aria-label="Main">
          <Tip content={theme === 'light' ? 'Dark theme' : 'Light theme'} side="bottom">
            <button type="button" className="tool" aria-label={theme === 'light' ? 'Switch to dark theme' : 'Switch to light theme'} onClick={() => setting('theme')(theme === 'light' ? 'dark' : 'light')}>
              <AnimatePresence mode="wait" initial={false}>
                <motion.span key={theme} className="icon-swap" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }} transition={{ duration: 0.2 }}>
                  {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
                </motion.span>
              </AnimatePresence>
            </button>
          </Tip>
          <button type="button" className="nav-btn" aria-label="Settings" onClick={() => setPanel('settings')}><GearSix size={18} /><span>Settings</span></button>
          <button type="button" className="nav-btn" aria-label="About" onClick={() => setPanel('about')}><Info size={18} /><span>About</span></button>
        </nav>
      </header>

      <main className="stage" inert={blocked}>
        <AnimatePresence>{setup && <Hero key="hero" onPlay={startGame} reduced={reduced} />}</AnimatePresence>

        {!setup && (
          <>
            <motion.div className="slot top-slot" {...enter(0.2, { y: -10 })}>
              <PlayerStrip name="ResNet" color={botColor} captured={botColor === 'w' ? byWhite : byBlack} lead={leadFor(botColor)} status={botStatus} reduced={reduced} active={phase === 'playing' && game.turn() === botColor} />
            </motion.div>
            <motion.aside className="panel analysis glass" aria-label="Analysis" {...enter(0.25, { x: -28 })}>
              <Analysis scores={scores} stats={stats} thinking={thinking} progress={progress} search={search} timer={timer} leader={leader} userColor={userColor} botColor={botColor} budget={BUDGET_MS} />
            </motion.aside>
          </>
        )}

        <motion.div className="board-area" layout={!reduced} transition={{ type: 'spring', bounce: 0.08, duration: 1.05 }}>
          <Aura ref={aura} enabled={settings.fx} reduced={reduced} theme={theme} />
          <Board
            pieces={pieces} orientation={orientation} color={userColor} interactive={interactive}
            selected={viewing ? null : selected} targets={targets} lastMove={lastMove} checkSquare={checkSquare}
            showCoords={settings.coords} promotion={promotion} ghosts={ghosts} fallenSquare={fallenSquare} reduced={reduced}
            onSelect={setSelected} onMove={playMove}
            onPromote={type => { const pending = promotion; setPromotion(null); playMove(pending.from, pending.to, type); }}
            onCancelPromotion={() => setPromotion(null)}
          >
            <FxLayer ref={fx} orientation={orientation} reduced={reduced} enabled={settings.fx} theme={theme} />
          </Board>
          <AnimatePresence>
            {showCard && <ResultCard key="result" result={result} side={side} moves={total} delay={result.reason === 'Checkmate' && !reduced ? 1.1 : 0.15} onNew={setupGame} onClose={() => setCard(false)} />}
          </AnimatePresence>
          <AnimatePresence>
            {setup && <motion.p key="caption" className="caption" initial={{ opacity: 0 }} animate={{ opacity: 1, transition: { delay: 1.2 } }} exit={{ opacity: 0 }}>One of my wins, February 2023</motion.p>}
          </AnimatePresence>
        </motion.div>

        {!setup && (
          <>
            <motion.aside className="panel moves-panel glass" aria-label="Moves" {...enter(0.3, { x: 28 })}>
              <MoveList history={history} view={view} onView={index => setView(index === null || index >= total ? null : index)} reduced={reduced} botColor={botColor} />
            </motion.aside>
            <motion.div className="slot bottom-slot" {...enter(0.35, { y: 10 })}>
              <PlayerStrip name="You" color={userColor} captured={userColor === 'w' ? byWhite : byBlack} lead={leadFor(userColor)} status={userStatus} reduced={reduced} active={phase === 'playing' && game.turn() === userColor} />
            </motion.div>
            <motion.div className="slot dock-slot" {...enter(0.45, { y: 20 })}>
              <Dock
                phase={phase} reduced={reduced} drawPending={drawPending}
                canDraw={phase === 'playing' && !thinking && !drawPending && !error}
                onNew={setupGame} onFlip={() => setFlipped(value => !value)} onDraw={offerDraw}
                onResign={() => finish({ winner: side === 'white' ? 'black' : 'white', reason: 'You resigned' })}
              />
            </motion.div>
          </>
        )}
      </main>

      <AnimatePresence>
        {flyers.map(flyer => <Flyer key={flyer.id} flyer={flyer} onDone={() => setFlyers(list => list.filter(item => item.id !== flyer.id))} />)}
      </AnimatePresence>

      <div className="sr-only" role="status" aria-live="polite">{setup ? '' : announcement}</div>

      <div className="toasts">
        <AnimatePresence>
          {error && (
            <motion.div key="error" className="toast error" role="alert" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
              <span>{error}</span>
              <button type="button" className="btn small" onClick={() => engineMove()}><ArrowClockwise size={15} />Retry</button>
            </motion.div>
          )}
          {notice && (
            <motion.div key="notice" className="toast" role="status" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}>
              <span>{notice}</span>
              <button type="button" className="tool" aria-label="Dismiss" onClick={() => setNotice(null)}><X size={15} /></button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {panel && (
          <Drawer key="drawer" tab={panel} onTab={setPanel} onClose={() => setPanel(null)} reduced={reduced}>
            {panel === 'settings' ? (
              <div className="settings">
                <Setting title="Thinking light" control={<Switch label="Thinking light" checked={settings.fx} onChange={setting('fx')} />}>Light streams around the board while ResNet searches, and trails follow each move.</Setting>
                <Setting title="Move preview" control={<Switch label="Move preview" checked={settings.ghost} onChange={setting('ghost')} />}>Faint pieces show the moves ResNet is weighing. Its favourite is the brightest.</Setting>
                <Setting title="Coordinates" control={<Switch label="Coordinates" checked={settings.coords} onChange={setting('coords')} />}>Files and ranks along the board edge.</Setting>
                <Setting title="Theme" control={
                  <div className="segmented" role="radiogroup" aria-label="Theme">
                    {['system', 'dark', 'light'].map(value => <button key={value} type="button" role="radio" aria-checked={settings.theme === value} onClick={() => setting('theme')(value)}>{value[0].toUpperCase() + value.slice(1)}</button>)}
                  </div>
                } />
              </div>
            ) : (
              <div className="about">
                <p className="lede">I’m Piyush Singh. ResNet plays chess the way I do.</p>
                <p>A 15-block residual network studied 3,451 of my games. For any position it estimates two things: how likely I am to play each move, and how good the position is.</p>
                <p>In openings I know well it plays one of my usual moves straight away. Otherwise it looks ahead for up to 3 seconds, usually about one. It then picks among the moves within a pawn of the best one, favoring the ones I’d be likeliest to choose. So it sometimes prefers a move that feels like mine over the strongest one.</p>
                <dl className="glossary">
                  <div><dt>Evaluation</dt><dd>Estimated advantage in pawns. Above zero favors White.</dd></div>
                  <div><dt>Instinct</dt><dd>How likely I am to play a move, before looking ahead.</dd></div>
                  <div><dt>Depth</dt><dd>How many moves ahead the search has looked.</dd></div>
                  <div><dt>Book</dt><dd>An opening move I have played here before, chosen without searching.</dd></div>
                  <div><dt>Positions</dt><dd>Positions checked during the search.</dd></div>
                </dl>
                <p className="note">Evaluations are estimates, and it can miss tactics.</p>
              </div>
            )}
          </Drawer>
        )}
      </AnimatePresence>
    </div>
  );
}
