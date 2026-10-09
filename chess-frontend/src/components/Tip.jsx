import { cloneElement, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { AnimatePresence, motion } from 'motion/react';

const EDGE = 8;

function Bubble({ id, anchor, content, side }) {
  const ref = useRef(null);

  useLayoutEffect(() => {
    const el = ref.current, target = anchor.current;
    if (!el || !target) return;
    const a = target.getBoundingClientRect(), b = el.getBoundingClientRect();
    const fitsAbove = a.top - b.height - 10 > EDGE;
    const under = side === 'bottom' ? a.bottom + b.height + 10 < innerHeight - EDGE || !fitsAbove : !fitsAbove;
    el.style.left = `${Math.min(Math.max(a.left + a.width / 2 - b.width / 2, EDGE), innerWidth - b.width - EDGE)}px`;
    el.style.top = `${under ? a.bottom + 10 : a.top - b.height - 10}px`;
    el.style.transformOrigin = under ? '50% 0' : '50% 100%';
  }, [anchor, side]);

  return (
    <motion.div
      ref={ref} id={id} role="tooltip" className="tip"
      initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, transition: { duration: 0.1 } }}
      transition={{ type: 'spring', stiffness: 520, damping: 34 }}
    >
      {content}
    </motion.div>
  );
}

/**
 * Explains the element it wraps on hover, keyboard focus or a long press. The child must be a
 * single DOM element; it gets the handlers and an aria-describedby pointing at the bubble.
 */
export default function Tip({ content, side = 'top', children }) {
  const anchor = useRef(null);
  const timer = useRef(0);
  const [open, setOpen] = useState(false);
  const id = useId();

  useEffect(() => () => clearTimeout(timer.current), []);
  useEffect(() => {
    if (!open) return undefined;
    const close = event => { if (event.type === 'scroll' || event.key === 'Escape') setOpen(false); };
    addEventListener('keydown', close);
    addEventListener('scroll', close, true);
    return () => { removeEventListener('keydown', close); removeEventListener('scroll', close, true); };
  }, [open]);

  if (!content) return children;
  const later = (value, ms) => { clearTimeout(timer.current); timer.current = setTimeout(() => setOpen(value), ms); };
  const own = children.props;

  return (
    <>
      {cloneElement(children, {
        ref: anchor,
        'aria-describedby': open ? id : undefined,
        onPointerEnter: event => { own.onPointerEnter?.(event); if (event.pointerType === 'mouse') later(true, 320); },
        onPointerLeave: event => { own.onPointerLeave?.(event); if (event.pointerType === 'mouse') later(false, 60); },
        onPointerDown: event => { own.onPointerDown?.(event); if (event.pointerType === 'mouse') later(false, 0); else later(true, 450); },
        onPointerUp: event => { own.onPointerUp?.(event); if (event.pointerType !== 'mouse') later(false, open ? 1800 : 0); },
        onFocus: event => { own.onFocus?.(event); if (event.target.matches(':focus-visible')) later(true, 120); },
        onBlur: event => { own.onBlur?.(event); later(false, 0); },
      })}
      {createPortal(<AnimatePresence>{open && <Bubble key="tip" id={id} anchor={anchor} content={content} side={side} />}</AnimatePresence>, document.body)}
    </>
  );
}
