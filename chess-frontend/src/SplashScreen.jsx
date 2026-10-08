import { useState } from 'react';

const SplashScreen = ({ onStart }) => {
    const [isExiting, setIsExiting] = useState(false);

    const handleStart = (side) => {
        setIsExiting(true);
        setTimeout(() => onStart(side), 750);
    };

    return (
        <div className={`fixed inset-0 z-[100] flex items-center justify-center bg-surface transition-opacity duration-700 ease-in-out ${isExiting ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>

            <div className={`flex flex-col items-center text-center max-w-lg w-full px-4 lg:px-6 transform transition-all duration-700 ease-out ${isExiting ? 'scale-95 opacity-0' : 'scale-100 opacity-100'}`}>

                <div className="mb-6 lg:mb-10">
                    <h1 className="text-xl lg:text-3xl font-semibold text-text-primary tracking-tight mb-1 lg:mb-2">
                        Chess Bot v2
                    </h1>
                    <p className="text-text-muted text-xs lg:text-sm">
                        Neural Chess Clone
                    </p>
                </div>

                <div className="flex flex-col lg:flex-row items-center justify-center gap-3 lg:gap-8 mb-8 lg:mb-12 text-center">
                    <div>
                        <div className="text-lg lg:text-2xl font-semibold text-text-primary font-mono">
                            Personal
                        </div>
                        <div className="text-[10px] lg:text-[11px] text-text-muted uppercase tracking-wide mt-0.5">Play style</div>
                    </div>

                    <div className="w-16 lg:w-px h-px lg:h-8 bg-surface-200" />

                    <div>
                        <div className="text-lg lg:text-2xl font-semibold text-text-primary font-mono">
                            Neural
                        </div>
                        <div className="text-[10px] lg:text-[11px] text-text-muted uppercase tracking-wide mt-0.5">Move search</div>
                    </div>

                    <div className="w-16 lg:w-px h-px lg:h-8 bg-surface-200" />

                    <div>
                        <div className="text-lg lg:text-2xl font-semibold text-accent font-mono">
                            Estimated
                        </div>
                        <div className="text-[10px] lg:text-[11px] text-text-muted uppercase tracking-wide mt-0.5">Scores</div>
                    </div>
                </div>

                <div className="flex gap-3 w-full max-w-xs lg:max-w-sm">
                    <button
                        onClick={() => handleStart('white')}
                        className="flex-1 bg-text-primary text-surface font-medium py-3.5 rounded-xl transition-all lg:hover:opacity-90 active:scale-[0.98]"
                    >
                        Play White
                    </button>
                    <button
                        onClick={() => handleStart('black')}
                        className="flex-1 bg-surface-100 text-text-primary font-medium py-3.5 rounded-xl border border-surface-200 transition-all lg:hover:bg-surface-300 active:scale-[0.98]"
                    >
                        Play Black
                    </button>
                </div>

            </div>
        </div>
    );
};

export default SplashScreen;
