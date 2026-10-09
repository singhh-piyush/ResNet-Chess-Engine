import { useEffect, useRef } from 'react';

export const FLIGHT_DELAY = 0.22; // the capturing piece arrives first
export const FLIGHT = 0.78;
const HANDOFF = 0.82;             // share of the flight after which the flyer fades into the tray
export const ARRIVAL = FLIGHT_DELAY + FLIGHT * HANDOFF;

const easeInOut = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const smooth = t => t * t * (3 - 2 * t);

/**
 * A captured piece that lifts off its square and glides along a soft arc into the capturer's
 * tray. It reads the tray slot's position every frame, so it lands exactly where the piece
 * appears even while the tray slides to make room, and it shrinks to the slot's size and
 * cross-fades into it instead of popping.
 */
export default function Flyer({ flyer, onDone }) {
  const img = useRef(null);
  const done = useRef(onDone);

  useEffect(() => { done.current = onDone; }, [onDone]);

  useEffect(() => {
    const el = img.current;
    const { x, y, size, slot } = flyer;
    const end = (slot.offsetWidth || size * 0.35) / size;
    let target = slot.getBoundingClientRect();
    const t0 = performance.now() + FLIGHT_DELAY * 1000;
    let raf = 0;
    const tick = (now) => {
      const t = Math.min(1, Math.max(0, (now - t0) / (FLIGHT * 1000)));
      if (slot.isConnected) target = slot.getBoundingClientRect();
      const tx = target.left + target.width / 2 - x, ty = target.top + target.height / 2 - y;
      const p = easeInOut(t);
      // Quadratic arc with its control point lifted above the midpoint: a gentle toss.
      const lift = -Math.min(90, Math.hypot(tx, ty) * 0.22);
      const cx = tx / 2, cy = ty / 2 + lift;
      const px = 2 * (1 - p) * p * cx + p * p * tx;
      const py = 2 * (1 - p) * p * cy + p * p * ty;
      const scale = (1 + (end - 1) * p) * (1 + 0.06 * Math.sin(Math.PI * p));
      el.style.transform = `translate3d(${px}px, ${py}px, 0) rotate(${-8 * Math.sin(Math.PI * p)}deg) scale(${scale})`;
      el.style.opacity = now < t0 ? 0 : 1 - smooth(Math.max(0, (t - HANDOFF) / (1 - HANDOFF)));
      if (t < 1) raf = requestAnimationFrame(tick); else done.current();
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [flyer]);

  return (
    <img
      ref={img} className="flyer" src={flyer.src} alt="" draggable={false}
      style={{ left: flyer.x - flyer.size / 2, top: flyer.y - flyer.size / 2, width: flyer.size, height: flyer.size, opacity: 0 }}
    />
  );
}
