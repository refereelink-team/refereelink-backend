import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const backendUrl = loadEnv(mode, process.cwd(), '').BACKEND_URL || 'http://localhost:8000';
  const backendWsUrl = backendUrl.replace(/^http/, 'ws');

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/api': backendUrl,
        '/ws': {
          target: backendWsUrl,
          ws: true,
        },
        '/video': backendUrl,
      },
    },
  };
});
