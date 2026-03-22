import { useState, useEffect, useRef, useMemo } from 'react';
import { Chess } from 'chess.js';
import { Chessboard } from 'react-chessboard';
import axios from 'axios';
import Tooltip from './components/Tooltip';
import SplashScreen from './SplashScreen';



const getPieceImg = (type, color) => {
  const names = {
    p: 'pawn',
    n: 'knight',
    b: 'bishop',
    r: 'rook',
    q: 'queen',
    k: 'king'
  };
  return `/TakenPiecesSVG/${names[type]}-${color}.svg`;
};

const getCapturedPieces = (game) => {
  const board = game.board();
  const currentCounts = {
    w: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 },
    b: { p: 0, n: 0, b: 0, r: 0, q: 0, k: 0 }
  };

  board.forEach(row => {
    row.forEach(square => {
      if (square) {
        currentCounts[square.color][square.type]++;
      }
    });
  });

  const captured = { w: [], b: [] };
  const STARTING_PIECES = { p: 8, n: 2, b: 2, r: 2, q: 1, k: 1 };

  ['q', 'r', 'b', 'n', 'p'].forEach(type => {
    const wMissing = STARTING_PIECES[type] - currentCounts['w'][type];
    for (let i = 0; i < wMissing; i++) captured['b'].push({ type, color: 'w' });

    const bMissing = STARTING_PIECES[type] - currentCounts['b'][type];
    for (let i = 0; i < bMissing; i++) captured['w'].push({ type, color: 'b' });
  });

  return captured;
};



const Modal = ({ children, onClose }) => (
  <div className="fixed inset-0 z-50 flex items-center justify-center bg-surface/90 backdrop-blur-sm" onClick={onClose}>
    <div className="bg-surface-50 border border-surface-200 p-6 lg:p-8 rounded-2xl max-w-md w-full mx-4 text-center relative overflow-hidden" onClick={e => e.stopPropagation()}>
      {children}
    </div>
  </div>
);

const InfoModal = ({ onClose }) => {
  const [isClosing, setIsClosing] = useState(false);
  const handleClose = () => {
    setIsClosing(true);
    setTimeout(onClose, 300);
  };

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-surface/80 backdrop-blur-sm pointer-events-auto" onClick={handleClose}>
      <div className={`h-full w-full max-w-md bg-surface-50 border-l border-surface-200 p-4 lg:p-8 relative overflow-y-auto ${isClosing ? 'animate-slide-out-right' : 'animate-slide-in-right'}`} onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6 border-b border-surface-200 pb-4">
          <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">About</h3>
          <button onClick={handleClose} className="text-text-muted lg:hover:text-text-primary transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="space-y-6 text-sm text-text-secondary leading-relaxed font-sans text-left">

          <div>
            <h4 className="text-accent font-medium mb-2">About This Engine</h4>
            <p>
              Most chess engines play perfect, algorithmic chess. <strong className="text-text-primary">This bot is an Imitation Model</strong> designed to replicate my playstyle.
            </p>
          </div>

          <div>
            <h4 className="text-accent font-medium mb-2">Architecture</h4>
            <p>
              Trained on <strong className="text-text-primary">3,437</strong> of my chess games (346,000+ positions). Powered by a custom <strong className="text-text-primary">Dual-Head SE-ResNet15</strong> neural network.
            </p>
          </div>

          <div>
            <h4 className="text-accent font-medium mb-2">How It Works</h4>
            <div className="space-y-3">
              <p>
                <strong className="text-text-primary relative inline-block">
                  <span className="relative z-10">The Policy Head (Instinct):</span>
                  <span className="absolute bottom-0.5 left-0 w-full h-2 bg-orange-500/20 -z-10 rounded-sm"></span>
                </strong><br />
                Mimics my intuition by filtering 30+ legal moves down to a handful of candidate moves that match learned behavioral patterns.
              </p>
              <p>
                <strong className="text-text-primary relative inline-block">
                  <span className="relative z-10">The Value Head (Calculation):</span>
                  <span className="absolute bottom-0.5 left-0 w-full h-2 bg-blue-500/20 -z-10 rounded-sm"></span>
                </strong><br />
                Simulates those specific candidate moves 1-ply into the future, evaluating the resulting board geometry from the opponent's perspective to prevent 1-move blunders.
              </p>
            </div>
          </div>

          <div>
            <h4 className="text-accent font-medium mb-2">Limitations</h4>
            <p>
              The bot only calculates 1 move ahead. It relies on instinct and short-range tactics, not deep algorithmic search. It is blind to forced mates and complex, multi-move defensive sequences.
            </p>
          </div>

          <div className="pt-4 border-t border-surface-200 mt-6 text-center">
            <p className="text-xs text-text-muted">
              Created by <a href="https://github.com/singhh-piyush" target="_blank" rel="noopener noreferrer" className="text-accent lg:hover:text-accent-light transition-colors font-medium">Piyush Singh</a>
            </p>
          </div>

        </div>
      </div>
    </div>
  );
};

const SettingsModal = ({ showGlow, setShowGlow, showAnalysis, setShowAnalysis, showHistory, setShowHistory, onClose }) => {
  const [isClosing, setIsClosing] = useState(false);
  const handleClose = () => {
    setIsClosing(true);
    setTimeout(onClose, 300);
  };

  const Toggle = ({ enabled, onChange }) => (
    <button
      onClick={onChange}
      className={`w-11 h-6 rounded-full flex items-center transition-colors p-1 ${enabled ? 'bg-accent justify-end' : 'bg-surface-200 justify-start'}`}
    >
      <div className="w-4 h-4 rounded-full bg-white" />
    </button>
  );

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-surface/80 backdrop-blur-sm pointer-events-auto" onClick={handleClose}>
      <div className={`h-full w-full max-w-sm bg-surface-50 border-l border-surface-200 p-4 lg:p-6 relative overflow-y-auto ${isClosing ? 'animate-slide-out-right' : 'animate-slide-in-right'}`} onClick={e => e.stopPropagation()}>

        <div className="flex items-center justify-between mb-6 border-b border-surface-200 pb-4">
          <h3 className="text-sm font-medium text-text-muted uppercase tracking-wider">Settings</h3>
          <button onClick={handleClose} className="text-text-muted lg:hover:text-text-primary transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="mb-8">
          <div className="text-[10px] text-text-muted uppercase mb-3 font-medium tracking-wider">Visuals</div>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-text-primary text-sm">Brain Glow</span>
              <Toggle enabled={showGlow} onChange={() => setShowGlow(!showGlow)} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-text-primary text-sm">Show Analysis Panel</span>
              <Toggle enabled={showAnalysis} onChange={() => setShowAnalysis(!showAnalysis)} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-text-primary text-sm">Show Move History</span>
              <Toggle enabled={showHistory} onChange={() => setShowHistory(!showHistory)} />
            </div>
          </div>
        </div>

        <div className="opacity-60 pointer-events-none">
          <div className="text-[10px] text-text-muted uppercase mb-4 font-medium tracking-wider flex items-center gap-2">
            Coming Soon
            <span className="text-[9px] bg-surface-100 px-1.5 py-0.5 rounded text-text-muted">Dev Preview</span>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-text-secondary text-sm">App Theme</span>
              <div className="flex bg-surface-100 border border-surface-200 rounded-lg p-1 gap-1">
                <button className="px-3 py-1 rounded bg-surface-200 text-text-secondary text-xs font-medium cursor-not-allowed">Dark</button>
                <button className="px-3 py-1 rounded text-text-muted text-xs font-medium cursor-not-allowed">Light</button>
              </div>
            </div>
            <p className="text-[11px] text-text-muted leading-tight">
              Switch between Dark and Light modes.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};

const Button = ({ onClick, children, variant = 'primary', className = '' }) => {
  const baseStyle = "px-5 py-2.5 rounded-xl font-medium transition-all duration-200";
  const variants = {
    primary: "bg-accent lg:hover:bg-accent-light text-black",
    secondary: "bg-surface-100 lg:hover:bg-surface-300 text-text-primary border border-surface-200"
  };
  return (
    <button onClick={onClick} className={`${baseStyle} ${variants[variant]} ${className}`}>
      {children}
    </button>
  );
};



const SanRenderer = ({ san, color }) => {
  const firstChar = san.charAt(0);
  const isPiece = ['N', 'B', 'R', 'Q', 'K'].includes(firstChar);

  const pieceTypeMap = { 'N': 'n', 'B': 'b', 'R': 'r', 'Q': 'q', 'K': 'k' };

  if (isPiece) {
    const pieceType = pieceTypeMap[firstChar];
    return (
      <span className="inline-flex items-center gap-1">
        <img
          src={getPieceImg(pieceType, color)}
          alt={firstChar}
          className={`w-3.5 h-3.5 select-none ${color === 'b' ? '[filter:drop-shadow(0.25px_0_0_white)_drop-shadow(-0.25px_0_0_white)_drop-shadow(0_0.25px_0_white)_drop-shadow(0_-0.25px_0_white)]' : ''}`}
        />
        <span>{san.slice(1)}</span>
      </span>
    );
  }

  return <span>{san}</span>;
};


const getMaterialScore = (capturedPieces) => {
  const values = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  return capturedPieces.reduce((acc, piece) => acc + (values[piece.type] || 0), 0);
};




export default function App() {
  const [game, setGame] = useState(new Chess());
  const [moveHistory, setMoveHistory] = useState([]);
  const [gameResult, setGameResult] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [playerSide, setPlayerSide] = useState('white');
  const [gameStatus, setGameStatus] = useState('SPLASH');

  const [showGlow, setShowGlow] = useState(true);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const [showHistory, setShowHistory] = useState(true);
  const [botEvaluation, setBotEvaluation] = useState(0);
  const [resignStreak, setResignStreak] = useState(0);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isInfoOpen, setIsInfoOpen] = useState(false);
  const [optionSquares, setOptionSquares] = useState({});

  const [botStats, setBotStats] = useState({
    confidence: 0,
    evaluation: 0,
    candidates: [],
    lastMove: '',
    bgLog: []
  });


  const [thinkingLog, setThinkingLog] = useState([]);
  const thinkingInterval = useRef(null);
  const moveListRef = useRef(null);

  useEffect(() => {
    if (moveListRef.current) {
      moveListRef.current.scrollTop = moveListRef.current.scrollHeight;
    }
  }, [moveHistory.length]);

  const captured = useMemo(() => getCapturedPieces(game), [game.fen()]);

  const playerCaptured = playerSide === 'white' ? captured['w'] : captured['b'];
  const botCaptured = playerSide === 'white' ? captured['b'] : captured['w'];


  const checkGameOver = (gameInstance) => {
    if (gameInstance.isGameOver()) {
      setTimeout(() => {
        setGameStatus('GAME_OVER');
        if (gameInstance.isCheckmate()) {
          let currentWinner = gameInstance.turn() === 'w' ? 'black' : 'white';
          setGameResult({ winner: currentWinner, reason: 'Checkmate' });
        } else if (gameInstance.isDraw()) {
          let reason = 'Draw';
          if (gameInstance.isStalemate()) reason = 'Stalemate';
          else if (gameInstance.isThreefoldRepetition()) reason = 'Repetition';
          else if (gameInstance.isInsufficientMaterial()) reason = 'Insufficient Material';
          setGameResult({ winner: 'draw', reason });
        }
      }, 1000);
      return true;
    }
    return false;
  };

  const safeGameMutate = (modify) => {
    setGame((g) => {
      const update = new Chess(g.fen());
      modify(update);
      return update;
    });
  };

  const startThinkingAnimation = () => {
    setThinkingLog(["Neural Net initialized.", "Scanning patterns..."]);
    let stage = 0;
    const stages = [
      "Scanning board state...",
      "Policy pruning top moves...",
      "Simulating futures...",
      "Value Head calculating...",
      "Selecting best move..."
    ];
    if (thinkingInterval.current) clearInterval(thinkingInterval.current);
    thinkingInterval.current = setInterval(() => {
      if (stage < stages.length) {
        setThinkingLog(prev => [...prev.slice(-5), stages[stage]]);
        stage++;
      }
    }, 800);
  };

  const stopThinkingAnimation = (finalLogs) => {
    if (thinkingInterval.current) clearInterval(thinkingInterval.current);
    if (finalLogs && finalLogs.length > 0) {
      setThinkingLog(finalLogs.slice(-6));
    } else {
      setThinkingLog(prev => [...prev, "Execution complete."]);
    }
  };

  const makeBotMove = async (currentFen) => {
    const startTime = Date.now();
    setIsThinking(true);
    startThinkingAnimation();

    try {
      const response = await axios.post('/predict', {
        fen: currentFen,
      });

      const { move, confidence, evaluation, candidates, thinking_log, is_fallback } = response.data;

      if (response.data.resign) {
        setGameResult({ winner: playerSide, reason: 'Resignation' });
        setGameStatus('GAME_OVER');
        setIsThinking(false);
        return;
      }

      if (response.data.game_over || !move) {
        stopThinkingAnimation(thinking_log);
        setIsThinking(false);
        return;
      }

      if (evaluation !== undefined) {
        setBotEvaluation(evaluation);
        const botLosing = playerSide === 'white' ? evaluation > 5.0 : evaluation < -5.0;
        if (botLosing) {
          setResignStreak(prev => {
            const newStreak = prev + 1;
            if (newStreak >= 3) {
              setTimeout(() => {
                setGameResult({ winner: playerSide, reason: 'Resignation' });
                setGameStatus('GAME_OVER');
              }, 500);
            }
            return newStreak;
          });
        } else {
          setResignStreak(0);
        }
      }

      setBotStats({
        confidence,
        evaluation: evaluation || 0,
        candidates: candidates || [],
        lastMove: move,
        bgLog: thinking_log || [],
      });

      const MINIMUM_THINK_TIME = 1500;
      const elapsed = Date.now() - startTime;
      if (elapsed < MINIMUM_THINK_TIME) {
        await new Promise(resolve => setTimeout(resolve, MINIMUM_THINK_TIME - elapsed));
      }

      stopThinkingAnimation(thinking_log);

      safeGameMutate((game) => {
        const from = move.substring(0, 2);
        const to = move.substring(2, 4);
        const promotion = move.length > 4 ? move.substring(4) : 'q';
        const result = game.move({ from, to, promotion });
        if (result) {
          setMoveHistory(prev => [...prev, result.san]);
        }
        checkGameOver(game);
      });

    } catch (err) {
      console.error("Bot Error:", err);
      setIsThinking(false);
    } finally {
      setIsThinking(false);
    }
  };

  // Handle legal move highlights
  const highlightMoves = (square) => {
    const piece = game.get(square);
    if (!piece || piece.color !== playerSide[0]) {
      setOptionSquares({});
      return;
    }
    const moves = game.moves({ square, verbose: true });
    if (moves.length === 0) {
      setOptionSquares({});
      return;
    }
    const newSquares = {};
    newSquares[square] = { background: 'rgba(255, 255, 0, 0.4)' };
    moves.forEach((move) => {
      const target = game.get(move.to);
      newSquares[move.to] = {
        background: target
          ? 'radial-gradient(transparent 0%, transparent 79%, rgba(0,0,0,.1) 80%)'
          : 'radial-gradient(circle, rgba(0,0,0,.1) 25%, transparent 25%)',
      };
    });
    setOptionSquares(newSquares);
  };

  const onSquareClick = (square) => {
    highlightMoves(square);
  };

  const onPieceDragBegin = (piece, sourceSquare) => {
    highlightMoves(sourceSquare);
  };

  const onDrop = (sourceSquare, targetSquare) => {
    setOptionSquares({});
    if (gameStatus !== 'PLAYING') return false;
    if (isThinking) return false;
    if (game.turn() !== playerSide[0]) return false;

    const gameCopy = new Chess(game.fen());
    try {
      const move = gameCopy.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: 'q',
      });
      if (move === null) return false;

      setGame(gameCopy);
      setMoveHistory(prev => [...prev, move.san]);

      if (!checkGameOver(gameCopy)) {
        makeBotMove(gameCopy.fen());
      }
      return true;
    } catch (e) {
      return false;
    }
  };

  const getConfidenceColor = (score) => {
    if (score >= 0.8) return 'bg-emerald-500';
    if (score >= 0.5) return 'bg-blue-500';
    return 'bg-amber-500';
  };

  const getConfidenceTextColor = (score) => {
    if (score >= 0.8) return 'text-emerald-500';
    if (score >= 0.5) return 'text-blue-500';
    return 'text-amber-500';
  };

  const startGame = (side) => {
    setPlayerSide(side);
    setGame(new Chess());
    setMoveHistory([]);
    setGameStatus('PLAYING');
    setGameResult(null);
    setThinkingLog([]);
    setBotStats({ confidence: 0, evaluation: 0, candidates: [], lastMove: '', bgLog: [] });
    setBotEvaluation(0);
    setResignStreak(0);
  };

  useEffect(() => {
    if (gameStatus === 'PLAYING' && playerSide === 'black' && game.turn() === 'w' && !isThinking && !game.isGameOver()) {
      makeBotMove(game.fen());
    }
  }, [gameStatus, playerSide, game]); // eslint-disable-line react-hooks/exhaustive-deps




  const [notification, setNotification] = useState(null);

  const handleResign = () => {
    setGameResult({ winner: playerSide === 'white' ? 'black' : 'white', reason: 'Resignation' });
    setGameStatus('GAME_OVER');
  };

  const handleOfferDraw = async () => {
    try {
      const response = await axios.post('/offer_draw', {
        fen: game.fen(),
        user_side: playerSide
      });

      const { accepted, message } = response.data;
      if (accepted) {
        setGameResult({ winner: 'draw', reason: 'Agreement' });
        setGameStatus('GAME_OVER');
      } else {
        setNotification({
          title: "Draw Declined",
          message: message || "The bot declined your draw offer.",
          type: "info"
        });
      }
    } catch (e) {
      console.error(e);
      setNotification({
        title: "Error",
        message: "Could not contact the bot.",
        type: "error"
      });
    }
  };



  return (
    <div className="min-h-screen bg-surface text-text-secondary font-sans flex flex-col overflow-hidden relative selection:bg-accent/30">

      <header className="h-14 bg-surface-50 border-b border-surface-200 flex items-center justify-between px-4 lg:px-6 z-40">
        <h1 className="text-base font-medium text-text-primary tracking-tight">
          Chess Bot v2
        </h1>

        <div className="flex items-center gap-1">
          <button
            onClick={() => { setIsInfoOpen(!isInfoOpen); setIsSettingsOpen(false); }}
            className={`p-2 rounded-lg text-text-muted lg:hover:text-text-primary lg:hover:bg-surface-100 transition-colors ${isInfoOpen ? 'bg-surface-100 text-text-primary' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </button>

          <button
            onClick={() => { setIsSettingsOpen(!isSettingsOpen); setIsInfoOpen(false); }}
            className={`p-2 rounded-lg text-text-muted lg:hover:text-text-primary lg:hover:bg-surface-100 transition-colors ${isSettingsOpen ? 'bg-surface-100 text-text-primary' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826 3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>

        {isSettingsOpen && (
          <SettingsModal
            showGlow={showGlow}
            setShowGlow={setShowGlow}
            showAnalysis={showAnalysis}
            setShowAnalysis={setShowAnalysis}
            showHistory={showHistory}
            setShowHistory={setShowHistory}
            onClose={() => setIsSettingsOpen(false)}
          />
        )}

        {isInfoOpen && (
          <InfoModal onClose={() => setIsInfoOpen(false)} />
        )}
      </header>

      {notification && (
        <Modal onClose={() => setNotification(null)}>
          <div className="animate-scale-in text-center">
            <h2 className="text-xl font-semibold mb-3 text-text-primary">
              {notification.title}
            </h2>
            <p className="text-text-muted mb-6 text-sm">
              {notification.message}
            </p>
            <Button onClick={() => setNotification(null)} variant="primary" className="w-full">
              Continue
            </Button>
          </div>
        </Modal>
      )}

      {gameStatus === 'SPLASH' && (
        <SplashScreen onStart={startGame} />
      )}

      <div className="flex-1 w-full flex flex-col items-center gap-6 p-4 px-6 overflow-y-auto h-auto lg:flex-row lg:justify-center lg:gap-12 lg:items-start lg:p-6 lg:px-12 lg:overflow-hidden lg:h-[800px]">

        <div className="w-full lg:w-auto lg:shrink-0 mt-0 lg:mt-10 order-3 lg:order-1">
          {showAnalysis && (
            <aside className="w-full lg:w-[450px] flex flex-col gap-5 max-h-[400px] lg:max-h-none lg:h-[720px] p-4 lg:p-6 lg:pl-0 overflow-y-auto lg:overflow-hidden transition-all duration-300">
              <div>
                <div className="text-sm text-text-muted uppercase tracking-wide font-medium mb-3">Analysis</div>
              </div>

              <Tooltip text="How well this matches my playstyle">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-text-secondary">
                    <span>Confidence</span>
                    <div className="flex items-center gap-2">
                      <span className={`font-medium ${getConfidenceTextColor(botStats.confidence)}`}>
                        {(botStats.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="h-1.5 bg-surface-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-700 ease-out ${getConfidenceColor(botStats.confidence)}`}
                      style={{ width: `${Math.max(5, botStats.confidence * 100)}%` }}
                    />
                  </div>
                </div>
              </Tooltip>

              <Tooltip text="Positive for White advantage, Negative for Black">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm text-text-secondary">
                    <span>Position Evaluation</span>
                    <span className={`font-medium ${(botStats.evaluation || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {(botStats.evaluation || 0) >= 0 ? '+' : ''}{(botStats.evaluation || 0).toFixed(2)}
                    </span>
                  </div>
                  <div className="h-2 bg-surface-200 rounded-full overflow-hidden relative">
                    <div className="absolute left-1/2 top-0 w-px h-full bg-surface-300 z-10" />
                    <div
                      className={`absolute top-0 h-full transition-all duration-700 ease-out rounded-full ${(botStats.evaluation || 0) >= 0
                        ? 'bg-gradient-to-r from-emerald-500/60 to-emerald-500'
                        : 'bg-gradient-to-l from-red-500/60 to-red-500'
                        }`}
                      style={{
                        left: (botStats.evaluation || 0) >= 0 ? '50%' : undefined,
                        right: (botStats.evaluation || 0) < 0 ? '50%' : undefined,
                        width: `${Math.min(50, Math.abs(botStats.evaluation || 0) * 10)}%`,
                      }}
                    />
                  </div>
                </div>
              </Tooltip>

              <div className="flex-1 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm text-text-muted uppercase tracking-wide font-medium">Candidates</span>
                  {isThinking && <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />}
                </div>

                <div className="flex-1 overflow-y-auto pr-1 custom-scrollbar flex flex-col gap-3">
                  {botStats.candidates.map((c, i) => {
                    let badgeColor = 'bg-surface-200 text-text-muted';
                    let badgeText = 'ANALYZED';
                    let tooltipText = "Analyzed by the Value Head";

                    if (c.status === 'SELECTED') {
                      badgeColor = 'bg-green-500/20 text-green-400';
                      badgeText = 'SELECTED';
                      tooltipText = "Chosen by the engine";
                    } else if (c.status === 'VETOED') {
                      badgeColor = 'bg-red-500/20 text-red-500';
                      badgeText = 'VETOED';
                      tooltipText = "Rejected due to low evaluation";
                    }

                    return (
                      <Tooltip key={i} text={tooltipText}>
                        <div className="p-4 rounded-lg flex flex-col gap-2 transition-colors bg-surface-100 lg:hover:bg-surface-200">
                          <div className="flex justify-between items-center">
                            <div className="flex items-center gap-2">
                              <span className="text-text-muted text-sm font-mono">{i + 1}.</span>
                              <span className="text-base text-text-primary font-medium">{c.san || c.move}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className={`text-[11px] px-2 py-0.5 rounded uppercase font-medium ${badgeColor}`}>
                                {badgeText}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center justify-between text-xs font-mono">
                            <span className="text-text-muted">
                              Instinct: <span className={c.status === 'SELECTED' ? 'text-accent' : 'text-text-secondary'}>{(c.confidence * 100).toFixed(1)}%</span>
                            </span>
                            {c.evaluation !== undefined && (
                              <span className={c.evaluation >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                                Eval: {c.evaluation >= 0 ? '+' : ''}{c.evaluation.toFixed(2)}
                              </span>
                            )}
                          </div>
                          <div className="w-full h-1.5 bg-surface-200 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all duration-300 ${(c.evaluation || 0) > 0 ? 'bg-emerald-500' : (c.evaluation || 0) < 0 ? 'bg-red-500' : 'bg-slate-500'
                                }`}
                              style={{ width: `${Math.min(100, Math.max(4, (((c.evaluation || 0) + 5.0) / 10.0) * 100))}%` }}
                            />
                          </div>
                        </div>
                      </Tooltip>
                    );
                  })}

                  {!isThinking && botStats.candidates.length === 0 && (
                    <div className="text-center text-text-muted text-sm py-10">
                      Waiting for turn...
                    </div>
                  )}

                  {isThinking && botStats.candidates.length === 0 && (
                    <div className="space-y-3 animate-pulse">
                      <div className="h-14 bg-surface-100 rounded-lg" />
                      <div className="h-14 bg-surface-100 rounded-lg" />
                      <div className="h-14 bg-surface-100 rounded-lg" />
                    </div>
                  )}
                </div>
              </div>
            </aside>
          )}
        </div>

        <div className="flex flex-col items-center mt-0 lg:mt-10 order-1 lg:order-2 w-full lg:w-auto lg:shrink-0">
          <div className="flex flex-col items-center gap-2 w-full lg:w-[600px] shrink-0">

            <div className="w-full flex items-center justify-start h-8 pl-1">
              <div className="flex items-center gap-0.5">
                {botCaptured.map((piece, i) => (
                  <img
                    key={i}
                    src={getPieceImg(piece.type, piece.color)}
                    alt={piece.type}
                    className={`w-6 h-6 ${piece.color === 'b' ? 'opacity-90 [filter:drop-shadow(0.25px_0_0_white)_drop-shadow(-0.25px_0_0_white)_drop-shadow(0_0.25px_0_white)_drop-shadow(0_-0.25px_0_white)]' : 'opacity-60'}`}
                  />
                ))}
                {(() => {
                  const botScore = getMaterialScore(botCaptured);
                  const playerScore = getMaterialScore(playerCaptured);
                  const diff = botScore - playerScore;
                  if (diff > 0) {
                    return <span className="text-text-muted text-xs font-medium ml-1">+{diff}</span>;
                  }
                  return null;
                })()}
              </div>
            </div>

            <div className="relative z-10 w-full aspect-square">
              <div className={`gemini-glow-effect layer-1 ${isThinking && showGlow ? 'active' : ''}`} />
              <div className={`gemini-glow-effect layer-2 ${isThinking && showGlow ? 'active' : ''}`} />
              <div className={`gemini-glow-effect layer-3 ${isThinking && showGlow ? 'active' : ''}`} />

              <div className="w-full h-full rounded shadow-2xl overflow-hidden border border-surface-200 bg-surface-50">
                <Chessboard
                  id="MainBoard"
                  position={game.fen()}
                  onPieceDrop={onDrop}
                  onSquareClick={onSquareClick}
                  onPieceDragBegin={onPieceDragBegin}
                  boardOrientation={playerSide}
                  customSquareStyles={optionSquares}
                  customDarkSquareStyle={{
                    backgroundImage: 'linear-gradient(135deg, #334155 0%, #1e293b 100%)',
                    boxShadow: 'inset 0 0 10px rgba(0,0,0,0.5)'
                  }}
                  customLightSquareStyle={{
                    backgroundImage: 'linear-gradient(135deg, #cbd5e1 0%, #94a3b8 100%)',
                    boxShadow: 'inset 0 0 5px rgba(0,0,0,0.2)'
                  }}
                  animationDuration={300}
                />
              </div>
            </div>

            <div className="w-full flex items-center justify-start h-8 pl-1">
              <div className="flex items-center gap-0.5">
                {playerCaptured.map((piece, i) => (
                  <img
                    key={i}
                    src={getPieceImg(piece.type, piece.color)}
                    alt={piece.type}
                    className={`w-6 h-6 ${piece.color === 'b' ? 'opacity-90 [filter:drop-shadow(0.25px_0_0_white)_drop-shadow(-0.25px_0_0_white)_drop-shadow(0_0.25px_0_white)_drop-shadow(0_-0.25px_0_white)]' : 'opacity-60'}`}
                  />
                ))}
                {(() => {
                  const botScore = getMaterialScore(botCaptured);
                  const playerScore = getMaterialScore(playerCaptured);
                  const diff = playerScore - botScore;
                  if (diff > 0) {
                    return <span className="text-text-muted text-xs font-medium ml-1">+{diff}</span>;
                  }
                  return null;
                })()}
              </div>
            </div>

            <div className={`flex gap-4 mt-4 h-12 items-center justify-center w-full transition-opacity duration-300 ${gameStatus === 'PLAYING' ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
              <button
                onClick={handleOfferDraw}
                className="flex-1 py-2 bg-surface-50 lg:hover:bg-surface-100 text-text-secondary border border-surface-200 rounded-xl font-medium text-sm transition-colors shadow-lg"
              >
                Offer Draw
              </button>
              <button
                onClick={handleResign}
                className="flex-1 py-2 bg-red-900/10 lg:hover:bg-red-900/20 text-red-400 border border-red-900/20 rounded-xl font-medium text-sm transition-colors shadow-lg"
              >
                Resign
              </button>
            </div>

          </div>
        </div>


        <div className="w-full lg:w-auto lg:shrink-0 mt-0 lg:mt-10 order-2 lg:order-3">
          {showHistory && (
            <aside className="w-full lg:w-[240px] flex flex-col max-h-[300px] lg:max-h-none lg:h-[720px] transition-all duration-300">
              <div className="p-5 pl-0">
                <div className="text-sm text-text-muted uppercase tracking-wide font-medium">Moves</div>
              </div>
              <div ref={moveListRef} className="flex-1 overflow-y-auto p-2 pl-0 custom-scrollbar">
                {(() => {
                  const history = moveHistory;
                  const moves = [];
                  for (let i = 0; i < history.length; i += 2) {
                    moves.push({
                      num: Math.floor(i / 2) + 1,
                      white: history[i],
                      black: history[i + 1] || '',
                      whiteIndex: i,
                      blackIndex: i + 1
                    });
                  }
                  if (moves.length === 0) return <div className="text-text-muted text-sm text-center mt-10">No moves yet</div>;

                  return moves.map((m, i) => (
                    <div key={i} className="grid grid-cols-[2rem_1fr_1fr] gap-1 items-center px-4 pl-0 py-1 lg:hover:bg-surface-100/50 rounded group">
                      <span className="text-text-muted text-sm font-mono lg:group-hover:text-text-secondary transition-colors">{m.num}.</span>
                      <span className={`text-left font-medium ${m.whiteIndex === history.length - 1 ? 'text-text-primary' : 'text-text-secondary'}`}>
                        <SanRenderer san={m.white} color="w" />
                      </span>
                      <span className={`text-left font-medium ${m.blackIndex === history.length - 1 ? 'text-text-primary' : 'text-text-secondary'}`}>
                        {m.black && <SanRenderer san={m.black} color="b" />}
                      </span>
                    </div>
                  ));
                })()}
              </div>
            </aside>
          )}
        </div>

      </div>


      {gameStatus === 'GAME_OVER' && gameResult && (
        <Modal onClose={() => setGameStatus('SPLASH')}>
          <div className="animate-scale-in text-center">
            <h2 className="text-2xl font-semibold mb-4 text-text-primary">
              {gameResult?.winner === playerSide ? 'Victory' : gameResult?.winner === 'draw' ? 'Draw' : 'Defeat'}
            </h2>

            <p className="text-text-muted mb-8 text-sm">
              {gameResult?.reason}
            </p>

            <div className="flex flex-col gap-3">
              <Button onClick={() => setGameStatus('SPLASH')} variant="primary" className="w-full">
                Play Again
              </Button>
              <button
                onClick={() => setGameStatus('REVIEW')}
                className="text-text-muted lg:hover:text-text-primary text-sm py-2 transition-colors"
              >
                View Board
              </button>
            </div>
          </div>
        </Modal>
      )}

      {gameStatus === 'REVIEW' && (
        <div className="fixed bottom-6 right-6 z-50 animate-scale-in">
          <Button onClick={() => setGameStatus('SPLASH')} variant="primary">
            New Game
          </Button>
        </div>
      )}


    </div>
  );
}
