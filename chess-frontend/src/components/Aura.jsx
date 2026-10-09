import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

const TAU = Math.PI * 2;
const POINTS = 48;
const SIZE = 220;            // internal canvas pixels; the element is 4.4 board-widths, scaled up by the GPU
const HALF = SIZE / 4.4 / 2; // the board's half-width in canvas pixels
const BLUR = 5;
const HUES = [200, 232, 264, 296, 200];

// Three overlapping blobs, each with its own size, colour offset and rhythm. While ResNet thinks,
// each also drifts on its own slow Lissajous path (drift: x and y reach, x and y rate, phase), so
// the layers slide past one another like liquid instead of breathing in place. Wave and drift
// rates are mutually irrational, so the motion never visibly repeats.
const LAYERS = [
  { scale: 1.12, alpha: 0.95, hue: 0, spin: 0.11, waves: [[2, 0.07, 0.31], [3, 0.05, -0.43], [5, 0.025, 0.6]], ripples: [[4, 0.035, 0.9], [7, 0.012, -1.3]], drift: [0.09, 0.07, 0.53, 0.41, 0] },
  { scale: 1.27, alpha: 0.65, hue: 18, spin: -0.08, waves: [[2, 0.08, -0.23], [3, 0.06, 0.37], [4, 0.03, -0.52]], ripples: [[5, 0.03, -0.8], [6, 0.015, 1.1]], drift: [0.13, 0.11, -0.37, 0.61, 2.1] },
  { scale: 1.45, alpha: 0.4, hue: 36, spin: 0.05, waves: [[2, 0.09, 0.17], [4, 0.05, -0.29], [3, 0.04, 0.41]], ripples: [[3, 0.04, 0.7], [5, 0.02, -1.0]], drift: [0.16, 0.14, 0.29, -0.47, 4.2] },
];

// Coloured masses that travel round the board's edge while ResNet thinks. They orbit in opposite
// directions at different rates, so they meet, merge into one glow and part again, and each swells
// and drifts in and out on its own rhythm (wobble). Idle, they sit tucked behind the board.
const ORBS = [
  { hue: 200, size: 0.62, orbit: 0.95, speed: 0.42, phase: 0, wobble: 0.83 },
  { hue: 236, size: 0.55, orbit: 1.05, speed: -0.31, phase: 1.3, wobble: 1.11 },
  { hue: 268, size: 0.7, orbit: 0.9, speed: 0.27, phase: 2.6, wobble: 0.67 },
  { hue: 302, size: 0.5, orbit: 1.1, speed: -0.47, phase: 3.9, wobble: 1.37 },
  { hue: 186, size: 0.58, orbit: 1, speed: 0.36, phase: 5.1, wobble: 0.94 },
];

// Part circle, part rounded square, so shapes hug the board without corners.
const squircle = theta => {
  const c = Math.cos(theta), s = Math.sin(theta);
  return [c * 0.45 + Math.sign(c) * Math.sqrt(Math.abs(c)) * 0.55, s * 0.45 + Math.sign(s) * Math.sqrt(Math.abs(s)) * 0.55];
};
const OUTLINE = Array.from({ length: POINTS }, (_, i) => [(i / POINTS) * TAU, ...squircle((i / POINTS) * TAU)]);

const ease = (value, target, rate, dt) => value + (target - value) * (1 - Math.exp(-rate * dt));
const wrap = angle => Math.atan2(Math.sin(angle), Math.cos(angle));

/**
 * A soft, colour-shifting blob behind the board. Idle, it barely glows and drifts. While ResNet
 * searches it comes alive: coloured masses travel round the board, merging and parting, the
 * outline morphs on a few-second rhythm, the colours swirl and the whole thing leans toward the
 * side of the board the search is looking at. Changes ease in over a second or two, so it flows
 * like liquid rather than pulsing.
 *
 * It lives inside the board's box, so it follows the board with no layout reads, and it is
 * drawn on a tiny canvas that is blurred at that size and scaled up, which keeps it cheap.
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
      raf: 0, last: 0, drawn: 0, t: 0, flow: 0, swirl: 0, hue: 0, light: false, points: [],
      act: 0, target: 0, glow: 0, focus: 0, focusTarget: 0, angle: -Math.PI / 2, angleTarget: -Math.PI / 2,
    };
    const c = SIZE / 2;

    const blob = (layer, morph, alpha) => {
      const spin = s.t * layer.spin;
      const pts = s.points;
      const [ax, ay, fx, fy, phase] = layer.drift;
      const reach = HALF * (0.25 + s.act * 1.25);
      const ox = c + reach * ax * Math.sin(s.t * fx + phase), oy = c + reach * ay * Math.sin(s.t * fy + phase * 1.3);
      for (let i = 0; i < POINTS; i++) {
        const [theta, ux, uy] = OUTLINE[i];
        let wave = 0;
        for (const [k, amp, speed] of layer.waves) wave += amp * Math.sin(k * (theta + spin) + s.t * speed);
        // Finer ripples that only exist while thinking: they travel round the edge like a current.
        if (s.act > 0.01) for (const [k, amp, speed] of layer.ripples) wave += s.act * amp * Math.sin(k * theta - s.t * speed + k);
        const lean = s.focus * 0.2 * Math.max(0, Math.cos(theta - s.angle)) ** 3;
        const radius = HALF * layer.scale * (1 + wave * morph + lean);
        pts[i] = [ox + ux * radius, oy + uy * radius];
      }
      ctx.beginPath();
      let mx = (pts[POINTS - 1][0] + pts[0][0]) / 2, my = (pts[POINTS - 1][1] + pts[0][1]) / 2;
      ctx.moveTo(mx, my);
      for (let i = 0; i < POINTS; i++) {
        const a = pts[i], b = pts[(i + 1) % POINTS];
        mx = (a[0] + b[0]) / 2; my = (a[1] + b[1]) / 2;
        ctx.quadraticCurveTo(a[0], a[1], mx, my);
      }
      const base = s.hue + layer.hue * (1 + s.act * 0.8); // layers fan further apart in colour while thinking
      const tone = s.light ? `${74 + s.act * 12}% ${62 - s.act * 4}%` : `${82 + s.act * 12}% ${56 + s.act * 4}%`;
      if (conic) {
        const fill = ctx.createConicGradient(spin * 1.6 + s.flow, ox, oy);
        HUES.forEach((hue, i) => fill.addColorStop(i / (HUES.length - 1), `hsl(${hue + base} ${tone})`));
        ctx.fillStyle = fill;
      } else ctx.fillStyle = `hsl(${HUES[1] + base} ${tone})`;
      ctx.globalAlpha = alpha;
      ctx.fill();
    };

    const orb = (item, alpha, tone) => {
      const angle = item.phase + s.swirl * item.speed;
      const [ux, uy] = squircle(angle);
      const dist = HALF * (item.orbit * (0.62 + s.act * 0.5) + 0.2 * Math.sin(s.t * item.wobble + item.phase));
      const x = c + ux * dist + Math.cos(s.angle) * s.focus * HALF * 0.3;
      const y = c + uy * dist + Math.sin(s.angle) * s.focus * HALF * 0.3;
      const r = HALF * item.size * (0.75 + s.act * 0.55) * (1 + 0.16 * Math.sin(s.t * item.wobble * 1.3 + item.phase * 2));
      const hue = item.hue + s.hue * 1.5;
      const fill = ctx.createRadialGradient(x, y, 0, x, y, r);
      fill.addColorStop(0, `hsl(${hue} ${tone} / 1)`);
      fill.addColorStop(0.5, `hsl(${hue} ${tone} / 0.55)`);
      fill.addColorStop(1, `hsl(${hue} ${tone} / 0)`);
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.fillRect(x - r, y - r, r * 2, r * 2);
    };

    const frame = (now) => {
      s.raf = requestAnimationFrame(frame);
      const busy = s.target || s.act > 0.02 || s.glow > 0.01;
      if (!busy && now - s.drawn < 33) return; // idle: 30 fps is plenty for a slow drift
      const dt = Math.min(0.05, (now - (s.last || now)) / 1000);
      s.last = now; s.drawn = now;
      // Everything eases over seconds, never snaps: the blob should feel like slow weather.
      s.act = ease(s.act, s.target, s.target ? 1.2 : 0.5, dt);
      s.focus = ease(s.focus, s.focusTarget * s.target, 0.9, dt);
      s.angle += wrap(s.angleTarget - s.angle) * (1 - Math.exp(-1.1 * dt));
      s.glow = ease(s.glow, 0, 0.6, dt);
      s.t += dt * (0.2 + s.act * 2.8);
      s.flow += dt * (0.03 + s.act * 0.5);
      s.swirl += dt * (0.1 + s.act * 1.2);
      s.hue = Math.sin(s.t * 0.09) * (18 + s.act * 16);

      const strength = (s.light ? 0.16 + s.act * 0.46 : 0.2 + s.act * 0.55) + s.glow * 0.08;
      ctx.clearRect(0, 0, SIZE, SIZE);
      if (filter) ctx.filter = `blur(${BLUR}px)`;
      ctx.globalCompositeOperation = s.light ? 'source-over' : 'lighter';
      for (let i = LAYERS.length - 1; i >= 0; i--) blob(LAYERS[i], 0.6 + s.act * 1.6, LAYERS[i].alpha * strength);
      const tone = s.light ? `${80 + s.act * 10}% 60%` : `${88 + s.act * 8}% ${58 + s.act * 4}%`;
      const orbAlpha = (s.light ? 0.08 + s.act * 0.42 : 0.1 + s.act * 0.55) + s.glow * 0.06;
      for (const item of ORBS) orb(item, orbAlpha, tone);
      ctx.globalCompositeOperation = 'source-over';
      ctx.globalAlpha = 1;
      if (filter) ctx.filter = 'none';
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
