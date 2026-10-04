/// <reference types="vitest/config" />
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

// 两个入口是本子项目的结构性隔离（公众包不含律师侧模块）；dev proxy 让开发期
// 与生产同源（后端没有 CORS，也不打算加）
export default defineConfig({
  plugins: [
    vue(),
    AutoImport({ resolvers: [ElementPlusResolver()] }),
    Components({ resolvers: [ElementPlusResolver()] }),
  ],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  build: {
    rollupOptions: {
      input: {
        public: fileURLToPath(new URL('./index.html', import.meta.url)),
        lawyer: fileURLToPath(new URL('./lawyer.html', import.meta.url)),
      },
    },
  },
  server: { proxy: { '/api': 'http://127.0.0.1:8138' } },
  test: {
    environment: 'jsdom',
    include: ['tests/**/*.spec.ts'],
    server: { deps: { inline: ['element-plus'] } },
  },
})
