import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/predict': 'http://127.0.0.1:7860', '/offer_draw': 'http://127.0.0.1:7860' } },
})
