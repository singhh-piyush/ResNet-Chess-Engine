/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
        
                surface: {
                    DEFAULT: '#0a0a0b',  
                    50: '#111113',       
                    100: '#18181b',      
                    200: '#1f1f23',      
                    300: '#27272a',       
                },
                accent: {
                    DEFAULT: '#b4c0d0',    // Light grey blue
                    light: '#d1d9e3',    
                    dark: '#93a3b8',       // Darker
                    muted: '#b4c0d020',    
                },
                text: {
                    primary: '#ffffff',
                    secondary: '#a1a1a6',
                    muted: '#5c5c63',
                }
            },
            fontFamily: {
                mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', "Liberation Mono", "Courier New", 'monospace'],
                sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'system-ui', 'sans-serif'],
            },
            keyframes: {
                'spin-slow': {
                    'from': { transform: 'rotate(0deg)' },
                    'to': { transform: 'rotate(360deg)' },
                },
                'breathe': {
                    '0%, 100%': { transform: 'scale(1)', opacity: '0.5' },
                    '50%': { transform: 'scale(1.05)', opacity: '1' },
                },
                'wavy-pulse': {
                    '0%, 100%': { transform: 'scaleY(1)', opacity: '0.4' },
                    '50%': { transform: 'scaleY(1.05)', opacity: '0.6' },
                },
                'slide-in-right': {
                    '0%': { transform: 'translateX(100%)', opacity: '0' },
                    '100%': { transform: 'translateX(0)', opacity: '1' },
                },
                'slide-out-right': {
                    '0%': { transform: 'translateX(0)', opacity: '1' },
                    '100%': { transform: 'translateX(100%)', opacity: '0' },
                },
                'scale-in': {
                    '0%': { transform: 'scale(0.95)', opacity: '0' },
                    '100%': { transform: 'scale(1)', opacity: '1' },
                }
            },
            animation: {
                'spin-slow': 'spin-slow 8s linear infinite',
                'breathe': 'breathe 4s ease-in-out infinite',
                'wavy-pulse': 'wavy-pulse 3s ease-in-out infinite',
                'slide-in-right': 'slide-in-right 0.3s ease-out forwards',
                'slide-out-right': 'slide-out-right 0.3s ease-in forwards',
                'scale-in': 'scale-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards', // Smooth pop-up
            }
        },
    },
    plugins: [],
}
