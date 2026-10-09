import { useEffect } from 'react';
import { animate, motion, useMotionValue, useTransform } from 'motion/react';

/** A number that eases to its new value instead of jumping. */
export default function Count({ value, decimals = 0, duration = 0.6, from = value, signed = false }) {
  const mv = useMotionValue(from);
  const text = useTransform(mv, v => (signed && v > 0.004 ? '+' : '') + (Math.abs(v) < 0.5 * 10 ** -decimals ? 0 : v).toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }));
  useEffect(() => {
    const controls = animate(mv, value, { duration, ease: [0.16, 1, 0.3, 1] });
    return () => controls.stop();
  }, [value, duration, mv]);
  return <motion.span>{text}</motion.span>;
}
