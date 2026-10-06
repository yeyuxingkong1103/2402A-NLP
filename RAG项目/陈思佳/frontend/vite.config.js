import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiProxy = {
  '/api': {
    target: 'http://localhost:8000',
    changeOrigin: true,
  },
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: apiProxy,
    watch: {
      ignored: ['**/*.stackdump'],
    },
  },
  preview: {
    port: 4173,
    proxy: apiProxy,
  },
})
