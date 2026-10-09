import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { X } from '@phosphor-icons/react';

const TABS = [['settings', 'Settings'], ['about', 'About']];
const FOCUSABLE = 'button:not(:disabled), [href], input, [tabindex]:not([tabindex="-1"])';

/** Side panel for settings and background. Traps focus, closes on Escape or the scrim. */
export default function Drawer({ tab, onTab, onClose, reduced, children }) {
  const panel = useRef(null);
  const close = useRef(onClose);
  useEffect(() => { close.current = onClose; }, [onClose]);

  useEffect(() => {
    const opener = document.activeElement;
    panel.current?.querySelector('[aria-selected="true"]')?.focus();
    const onKey = event => {
      if (event.key === 'Escape') { event.preventDefault(); close.current(); return; }
      if (event.key !== 'Tab' || !panel.current) return;
      const items = [...panel.current.querySelectorAll(FOCUSABLE)];
      const first = items[0], last = items.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('keydown', onKey); opener?.focus?.(); };
  }, []);

  const spring = reduced ? { duration: 0 } : { type: 'spring', stiffness: 300, damping: 34 };
  return (
    <>
      <motion.div className="scrim" onClick={() => close.current()} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: reduced ? 0 : 0.25 }} />
      <motion.aside
        ref={panel} className="drawer" role="dialog" aria-modal="true" aria-label={tab === 'about' ? 'About' : 'Settings'}
        initial={{ x: '105%' }} animate={{ x: 0 }} exit={{ x: '105%' }} transition={spring}
      >
        <div className="drawer-head">
          <div className="tabs" role="tablist">
            {TABS.map(([key, label]) => (
              <button key={key} type="button" role="tab" aria-selected={tab === key} tabIndex={tab === key ? 0 : -1} onClick={() => onTab(key)}
                onKeyDown={event => { if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') { const next = key === 'settings' ? 'about' : 'settings'; onTab(next); event.currentTarget.parentElement.querySelector(`[data-tab="${next}"]`)?.focus(); } }}
                data-tab={key}
              >
                {tab === key && <motion.span layoutId="tab-pill" className="pill" transition={spring} />}
                <span>{label}</span>
              </button>
            ))}
          </div>
          <button type="button" className="tool" aria-label="Close" onClick={() => close.current()}><X size={18} /></button>
        </div>
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={tab} className="drawer-body" role="tabpanel"
            initial={reduced ? false : { opacity: 0, x: tab === 'about' ? 16 : -16 }} animate={{ opacity: 1, x: 0 }} exit={reduced ? undefined : { opacity: 0, x: tab === 'about' ? -16 : 16 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            {children}
          </motion.div>
        </AnimatePresence>
      </motion.aside>
    </>
  );
}
