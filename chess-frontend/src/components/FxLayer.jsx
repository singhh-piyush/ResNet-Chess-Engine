import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { parseUci, squareToCell } from '../lib/chess';

function createStore() {
  return {
    w: 0, h: 0, last: 0, raf: 0,
    orientation: 'white', reduced: false, enabled: true,
    accent: '108 147 255', danger: '239 106 82',
    leads: [], trails: [], bursts: [], ripples: [], shocks: [],
    staticLead: null,
  };
}

const rgba = (rgb, a) => `rgb(${rgb} / ${Math.max(0, Math.min(1, a))})`;
const out = t => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);

/**
 * Canvas overlay for effects that move faster or more fluidly than the DOM should: the glow
 * under the squares the engine is weighing, light trails behind moving pieces, capture bursts
 * and shockwaves. It reads a plain mutable store, so React never
 * re-renders for animation, and it only runs while something is on screen.
 */
const FxLayer = forwardRef(function FxLayer({ orientation, reduced, enabled, theme }, ref) {
  const canvas = useRef(null);
  const store = useRef(null);
  if (store.current == null) store.current = createStore();

  useEffect(() => {
    const s = store.current;
    s.orientation = orientation;
    s.reduced = reduced;
    s.enabled = enabled;
    const css = getComputedStyle(document.documentElement);
    s.accent = css.getPropertyValue('--accent-rgb').trim() || s.accent;
    s.danger = css.getPropertyValue('--danger-rgb').trim() || s.danger;
    s.api?.redraw();
  }, [orientation, reduced, enabled, theme]);

  useEffect(() => {
    const s = store.current;
    const el = canvas.current;
    const ctx = el.getContext('2d');

    const center = (square) => {
      const { col, row } = squareToCell(square, s.orientation);
      const size = s.w / 8;
      return [(col + 0.5) * size, (row + 0.5) * size];
    };

    const drawStatic = () => {
      ctx.clearRect(0, 0, s.w, s.h);
      if (!s.staticLead) return;
      const { col, row } = squareToCell(s.staticLead, s.orientation);
      const size = s.w / 8;
      ctx.fillStyle = rgba(s.accent, 0.4);
      ctx.fillRect(col * size, row * size, size, size);
    };

    const busy = () => s.leads.length || s.trails.length || s.bursts.length || s.ripples.length || s.shocks.length;

    const frame = (now) => {
      const dt = Math.min(0.05, (now - (s.last || now)) / 1000);
      s.last = now;
      const size = s.w / 8;
      ctx.clearRect(0, 0, s.w, s.h);

      // Soft light under the square the engine currently favors; the previous one fades out.
      s.leads = s.leads.filter(lead => {
        lead.alpha += (lead.target - lead.alpha) * (1 - Math.exp(-dt * 2.2));
        if (lead.target === 0 && lead.alpha < 0.02) return false;
        const [x, y] = center(lead.square);
        const glow = ctx.createRadialGradient(x, y, size * 0.1, x, y, size * 0.95);
        glow.addColorStop(0, rgba(s.accent, lead.alpha * 0.5));
        glow.addColorStop(1, rgba(s.accent, 0));
        ctx.fillStyle = glow;
        ctx.fillRect(x - size, y - size, size * 2, size * 2);
        return true;
      });

      // Light trail behind a moving piece: a tapered streak whose tail catches up with its head.
      s.trails = s.trails.filter(trail => {
        const t = (now - trail.t0) / 620;
        if (t >= 1) return false;
        const [ax, ay] = center(trail.from);
        const [bx, by] = center(trail.to);
        const head = out(t / 0.55), tail = out((t - 0.12) / 0.88);
        const hx = ax + (bx - ax) * head, hy = ay + (by - ay) * head;
        const tx = ax + (bx - ax) * tail, ty = ay + (by - ay) * tail;
        const len = Math.hypot(hx - tx, hy - ty);
        if (len < 1) return true;
        const nx = -(hy - ty) / len, ny = (hx - tx) / len, wide = size * 0.17 * trail.strength;
        const fill = ctx.createLinearGradient(tx, ty, hx, hy);
        fill.addColorStop(0, rgba(trail.rgb, 0));
        fill.addColorStop(1, rgba(trail.rgb, 0.55 * trail.strength));
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(hx + nx * wide, hy + ny * wide);
        ctx.arc(hx, hy, wide, Math.atan2(ny, nx), Math.atan2(ny, nx) + Math.PI, true);
        ctx.closePath(); ctx.fill();
        return true;
      });

      // Capture bursts: shards thrown out of the square, slowing as they fade.
      s.bursts = s.bursts.filter(burst => {
        burst.life -= dt;
        if (burst.life <= 0) return false;
        const k = burst.life / burst.total;
        for (const p of burst.particles) {
          p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 0.93; p.vy *= 0.93; p.spin += p.vs * dt;
          ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.spin);
          ctx.fillStyle = rgba(p.rgb, k);
          ctx.fillRect(-p.r, -p.r * 0.45, p.r * 2 * k + 1, p.r * 0.9);
          ctx.restore();
        }
        return true;
      });

      // Landing ripples.
      s.ripples = s.ripples.filter(ripple => {
        const t = (now - ripple.t0) / 600;
        if (t >= 1) return false;
        const [x, y] = center(ripple.square);
        ctx.beginPath(); ctx.arc(x, y, size * (0.2 + 0.6 * out(t)), 0, Math.PI * 2);
        ctx.strokeStyle = rgba(s.accent, (1 - t) * 0.7); ctx.lineWidth = 2 * (1 - t) + 0.5; ctx.stroke();
        return true;
      });

      // Check and mate shockwaves.
      s.shocks = s.shocks.filter(shock => {
        const t = (now - shock.t0) / shock.duration;
        if (t >= 1) return false;
        const [x, y] = center(shock.square);
        ctx.beginPath(); ctx.arc(x, y, shock.reach * out(t), 0, Math.PI * 2);
        ctx.strokeStyle = rgba(s.danger, (1 - t) * 0.85); ctx.lineWidth = size * shock.weight * (1 - t) + 1; ctx.stroke();
        return true;
      });

      if (busy()) s.raf = requestAnimationFrame(frame); else { s.raf = 0; s.last = 0; ctx.clearRect(0, 0, s.w, s.h); }
    };

    const run = () => { if (!s.raf && !s.reduced) s.raf = requestAnimationFrame(frame); };

    const resize = () => {
      const rect = el.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      s.w = rect.width; s.h = rect.height;
      el.width = Math.round(rect.width * dpr); el.height = Math.round(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (s.reduced) drawStatic();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(el);
    resize();

    // Each square the search favours keeps a glow, weighted by how highly it ranks now.
    // Nothing fades until the decision, so the options stay visible while it weighs them.
    const lead = (square, weight = 1) => {
      if (s.reduced) { if (weight >= 1) { s.staticLead = square; drawStatic(); } return; }
      const current = s.leads.find(item => item.square === square);
      if (current) current.target = weight; else s.leads.push({ square, alpha: 0, target: weight });
      run();
    };

    s.api = {
      redraw() { if (s.reduced) drawStatic(); },
      lead(square, weight) { if (s.enabled) lead(square, weight); },
      decide(uci) {
        if (!s.enabled) return;
        const { to } = parseUci(uci);
        for (const item of s.leads) if (item.square !== to) item.target = 0;
        lead(to, 1);
      },
      stop() {
        for (const item of s.leads) item.target = 0;
        if (s.reduced) { s.staticLead = null; drawStatic(); }
        run();
      },
      clear() {
        s.leads = []; s.staticLead = null;
        if (!s.raf) ctx.clearRect(0, 0, s.w, s.h);
      },
      trail(from, to, strong) {
        if (s.reduced) return;
        s.trails.push({ from, to, t0: performance.now(), rgb: strong ? s.accent : '255 255 255', strength: strong ? 1 : 0.6 });
        run();
      },
      land(square) {
        if (s.reduced) return;
        s.ripples.push({ square, t0: performance.now() }); run();
      },
      burst(square, color) {
        if (s.reduced) return;
        const [x, y] = center(square);
        const rgb = color === 'w' ? '240 242 246' : '28 31 40';
        const particles = Array.from({ length: 18 }, (_, i) => {
          const angle = (i / 18) * Math.PI * 2 + Math.random() * 0.5;
          const speed = 70 + Math.random() * 150;
          return { x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, r: 2 + Math.random() * 3, spin: angle, vs: (Math.random() - 0.5) * 12, rgb: i % 3 === 0 ? s.accent : rgb };
        });
        s.bursts.push({ particles, life: 0.75, total: 0.75 }); run();
      },
      shock(square, mate) {
        if (s.reduced) return;
        const size = s.w / 8;
        s.shocks.push(mate
          ? { square, t0: performance.now(), duration: 1100, reach: s.w * 1.15, weight: 0.5 }
          : { square, t0: performance.now(), duration: 650, reach: size * 1.4, weight: 0.18 });
        run();
      },
    };

    return () => {
      observer.disconnect();
      cancelAnimationFrame(s.raf);
      s.raf = 0;
    };
  }, []);

  useImperativeHandle(ref, () => ({
    lead: (square, weight) => store.current.api?.lead(square, weight),
    decide: uci => store.current.api?.decide(uci),
    stop: () => store.current.api?.stop(),
    clear: () => store.current.api?.clear(),
    trail: (from, to, strong) => store.current.api?.trail(from, to, strong),
    land: square => store.current.api?.land(square),
    burst: (square, color) => store.current.api?.burst(square, color),
    shock: (square, mate) => store.current.api?.shock(square, mate),
  }), []);

  return <canvas ref={canvas} className="fx" aria-hidden="true" />;
});

export default FxLayer;
