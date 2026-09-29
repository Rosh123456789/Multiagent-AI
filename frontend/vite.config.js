import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Forward /workflow and /health to the FastAPI backend.
      '/workflow': { target: 'http://localhost:8000', changeOrigin: true },
      '/health':   { target: 'http://localhost:8000', changeOrigin: true },
      '/demo':     { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
});
