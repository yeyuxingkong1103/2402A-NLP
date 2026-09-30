import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // 精排首次加载 / LLM 生成可能较慢，放宽代理超时避免 502
        timeout: 300000,
        proxyTimeout: 300000,
      },
    },
  },
})
