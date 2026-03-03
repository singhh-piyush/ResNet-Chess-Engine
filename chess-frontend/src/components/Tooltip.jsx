import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';

export default function Tooltip({ text, children }) {
    const [isVisible, setIsVisible] = useState(false);
    const [coords, setCoords] = useState({ top: 0, left: 0 });
    const triggerRef = useRef(null);

    const handleMouseEnter = () => {
        if (triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setCoords({
                top: rect.top - 8,
                left: rect.left + rect.width / 2
            });
            setIsVisible(true);
        }
    };

    const handleMouseLeave = () => {
        setIsVisible(false);
    };

    return (
        <>
            <div
                ref={triggerRef}
                className="relative w-full"
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
            >
                {children}
            </div>

            {isVisible && createPortal(
                <div
                    className="fixed z-[9999] pointer-events-none flex flex-col items-center"
                    style={{
                        top: coords.top,
                        left: coords.left,
                        transform: 'translate(-50%, -100%)'
                    }}
                >
                    <div className="bg-black/80 backdrop-blur-md text-white/90 text-sm px-3 py-2 rounded-xl shadow-2xl shadow-black/50 max-w-[220px] text-center animate-in fade-in zoom-in-95 duration-200">
                        {text}
                    </div>
                    <div className="w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-black/80"></div>
                </div>,
                document.body
            )}
        </>
    );
}
