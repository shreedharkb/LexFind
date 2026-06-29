import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// LexFind – Vite Configuration
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['lucide-react'],
  },
  server: {
    proxy: {
      // Proxy /api/* to the FastAPI backend during local dev.
      // This avoids CORS issues and makes blob downloads same-origin.
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
