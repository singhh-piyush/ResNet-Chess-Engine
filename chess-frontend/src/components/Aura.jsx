import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

const TAU = Math.PI * 2;
const SPAN = 1.8;            // the canvas covers 1.8 board-widths, centred on the board
const SIZE = 240;            // internal pixels; the GPU scales the canvas up to its CSS size
const BOARD = SIZE / SPAN;   // the board's width in canvas pixels

// Six soft shapes, each a little larger than the board, filled with a sweep of blue, violet,
// magenta and cyan. Their corners morph, they rock and drift, and because the board covers
// their middle only slivers of colour peek out past its edges, in different places as they move.
// grow: how far each side reaches past the board; blur, alpha and period set its softness,
// strength and rhythm; from rotates its colour sweep.
const SHAPES = [
  { grow: 0.03, blur: 0.04, alpha: 0.5, period: 6, from: 0, colors: ['#3b82f6', '#8b5cf6', '#d946ef', '#06b6d4'] },
  { grow: 0.03, blur: 0.04, alpha: 0.5, period: 8, from: Math.PI, colors: ['#06b6d4', '#d946ef', '#8b5cf6', '#3b82f6'] },
  { grow: 0.05, blur: 0.047, alpha: 0.38, period: 7, from: Math.PI / 2, colors: ['#8b5cf6', '#d946ef', '#ec4899', '#3b82f6'] },
  { grow: 0.05, blur: 0.047, alpha: 0.38, period: 9, from: Math.PI * 1.5, colors: ['#06b6d4', '#3b82f6', '#8b5cf6'] },
  { grow: 0.07, blur: 0.054, alpha: 0.28, period: 10, from: Math.PI / 4, colors: ['#3b82f6', '#06b6d4', '#8b5cf6'] },
  { grow: 0.07, blur: 0.054, alpha: 0.28, period: 12, from: Math.PI * 1.25, colors: ['#d946ef', '#8b5cf6', '#3b82f6'] },
].map((shape, i) => ({
  ...shape,
  seed: i * 2.39,
  // Each corner radius morphs at its own rate, so the outline never settles into a repeating loop.
  rates: Array.from({ length: 8 }, (_, j) => 0.7 + 0.6 * ((j * 0.618 + i * 0.31) % 1)),
}));

const ease = (value, target, rate, dt) => value + (target - value) * (1 - Math.exp(-rate * dt));
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));

/** A rectangle with elliptical corners, like CSS border-radius with eight values (fractions of w and h). */
function morphRect(ctx, w, h, r) {
  const fit = Math.min(1, 1 / (r[0] + r[1]), 1 / (r[3] + r[2]), 1 / (r[4] + r[7]), 1 / (r[5] + r[6]));
  const [tl, tr, br, bl] = [r[0] * fit * w, r[1] * fit * w, r[2] * fit * w, r[3] * fit * w];
  const [tlv, trv, brv, blv] = [r[4] * fit * h, r[5] * fit * h, r[6] * fit * h, r[7] * fit * h];
  const x = -w / 2, y = -h / 2;
  ctx.beginPath();
  ctx.moveTo(x + tl, y);
  ctx.lineTo(x + w - tr, y);
  ctx.ellipse(x + w - tr, y + trv, tr, trv, 0, -Math.PI / 2, 0);
  ctx.lineTo(x + w, y + h - brv);
  ctx.ellipse(x + w - br, y + h - brv, br, brv, 0, 0, Math.PI / 2);
  ctx.lineTo(x + bl, y + h);
  ctx.ellipse(x + bl, y + h - blv, bl, blv, 0, Math.PI / 2, Math.PI);
  ctx.lineTo(x, y + tlv);
  ctx.ellipse(x + tl, y + tlv, tl, tlv, 0, Math.PI, Math.PI * 1.5);
  ctx.closePath();
}

/**
 * A colour-shifting blob behind the board. Idle, it is a faint, slow drift. While ResNet
 * searches it brightens and speeds up, its shapes rocking and morphing so colour slips out
 * from different edges, and it leans toward the side of the board the search is looking at.
 *
 * It lives inside the board's box, so it follows the board with no layout reads, and it is
 * drawn on one small canvas that is blurred at that size and scaled up, which keeps it cheap.
 */
const Aura = forwardRef(function Aura({ enabled, reduced, theme }, ref) {
  const canvas = useRef(null);
  const control = useRef(null);
  const config = useRef({ enabled, reduced });

  useEffect(() => {
    config.current = { enabled, reduced };
    control.current?.refresh();
  }, [enabled, reduced, theme]);

  useEffect(() => {
    const el = canvas.current;
    el.width = el.height = SIZE;
    const ctx = el.getContext('2d');
    const conic = typeof ctx.createConicGradient === 'function';
    const filter = 'filter' in ctx;
    el.classList.toggle('soft', !filter);
    const s = {
      raf: 0, last: 0, drawn: 0, t: 0, flow: 0, light: false,
      act: 0, target: 0, glow: 0, focus: 0, focusTarget: 0, angle: -Math.PI / 2, angleTarget: -Math.PI / 2,
    };
    const c = SIZE / 2;
    const radii = new Array(8);

    const shape = (item, strength) => {
      const w = TAU / item.period, t = s.t, k = item.seed;
      const rotate = 0.16 * Math.sin(w * t + k) + 0.05 * Math.sin(2.3 * w * t + k * 1.7);
      const lean = s.focus * 0.035 * BOARD;
      const x = c + 0.032 * BOARD * Math.sin(0.9 * w * t + k * 2.1) + Math.cos(s.angle) * lean;
      const y = c + 0.032 * BOARD * Math.sin(1.13 * w * t + k * 0.6 + 1) + Math.sin(s.angle) * lean;
      for (let j = 0; j < 8; j++) radii[j] = 0.5 + 0.2 * Math.sin(w * item.rates[j] * t + k + j * 1.37);
      const side = BOARD * (1 + item.grow * 2) * (0.96 + s.act * 0.04);
      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(rotate);
      if (filter) ctx.filter = `blur(${(item.blur * BOARD).toFixed(1)}px)`;
      if (conic) {
        const fill = ctx.createConicGradient(item.from + s.flow, 0, 0);
        item.colors.forEach((color, i) => fill.addColorStop(i / item.colors.length, color));
        fill.addColorStop(1, item.colors[0]);
        ctx.fillStyle = fill;
      } else ctx.fillStyle = item.colors[1];
      ctx.globalAlpha = item.alpha * strength;
      morphRect(ctx, side, side, radii);
      ctx.fill();
      ctx.restore();
    };

    const frame = (now) => {
      s.raf = requestAnimationFrame(frame);
      const busy = s.target || s.act > 0.02 || s.glow > 0.01;
      if (!busy && now - s.drawn < 33) return; // idle: 30 fps is plenty for a slow drift
      const dt = Math.min(0.05, (now - (s.last || now)) / 1000);
      s.last = now; s.drawn = now;
      // Activity eases in and out over a second or two; the shapes speed up with it, never snap.
      s.act = ease(s.act, s.target, s.target ? 1.1 : 0.6, dt);
      s.focus = ease(s.focus, s.focusTarget * s.target, 0.9, dt);
      s.angle += wrap(s.angleTarget - s.angle) * (1 - Math.exp(-1.1 * dt));
      s.glow = ease(s.glow, 0, 0.6, dt);
      s.t += dt * (0.45 + s.act * 0.55);
      s.flow += dt * (0.04 + s.act * 0.16);

      const strength = (s.light ? 0.16 + s.act * 0.7 : 0.2 + s.act * 0.8) + s.glow * 0.1;
      ctx.clearRect(0, 0, SIZE, SIZE);
      for (let i = SHAPES.length - 1; i >= 0; i--) shape(SHAPES[i], strength);
      ctx.globalAlpha = 1;
    };

    const run = () => {
      const { enabled: on, reduced: still } = config.current;
      if (!on || still || s.raf || document.hidden) return;
      s.last = 0;
      s.raf = requestAnimationFrame(frame);
    };
    const halt = () => { cancelAnimationFrame(s.raf); s.raf = 0; };
    const visibility = () => (document.hidden ? halt() : run());
    document.addEventListener('visibilitychange', visibility);

    control.current = {
      refresh() {
        s.light = document.documentElement.dataset.theme === 'light';
        if (!config.current.enabled || config.current.reduced) { halt(); ctx.clearRect(0, 0, SIZE, SIZE); } else run();
      },
      start() { s.target = 1; s.focusTarget = 0; },
      attend(angle, strength = 0.7) { s.angleTarget = angle; s.focusTarget = Math.max(s.focusTarget * 0.6, strength); },
      decide() { s.glow = 1; s.focusTarget = 1; },
      stop() { s.target = 0; s.focusTarget = 0; },
      clear() { s.target = 0; s.act = 0; s.glow = 0; s.focus = 0; s.focusTarget = 0; },
    };
    control.current.refresh();

    return () => {
      document.removeEventListener('visibilitychange', visibility);
      halt();
    };
  }, []);

  useImperativeHandle(ref, () => ({
    start: () => control.current?.start(),
    attend: (angle, strength) => control.current?.attend(angle, strength),
    decide: () => control.current?.decide(),
    stop: () => control.current?.stop(),
    clear: () => control.current?.clear(),
  }), []);

  return <canvas ref={canvas} className="aura" aria-hidden="true" />;
});

export default Aura;
