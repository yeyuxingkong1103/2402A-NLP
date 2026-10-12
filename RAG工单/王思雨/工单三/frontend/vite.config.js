// 工单编号：人工智能NLP-RAG-PDF文档的表格解析及检索优化
// 文件说明：Vite 配置。开发服务器 5173，代理 /api 到 FastAPI 8000，避免前端处理跨域。
import { defineConfig } from 'vite'          // Vite 配置函数
import vue from '@vitejs/plugin-vue'         // Vue 单文件组件支持

export default defineConfig({
  plugins: [vue()],                          // 启用 Vue 插件
  server: {
    port: 5173,                              // 开发服务器端口
    proxy: {
      // 后端接口按工单要求挂在 /ask、/health 等根路径上，不含 /api 前缀，
      // 所以代理时必须把 /api 前缀去掉，否则后端收到 /api/ask 会返回 404。
      '/api': {
        target: 'http://127.0.0.1:8000',      // 后端地址
        changeOrigin: true,                   // 改写出站请求的 Host
        rewrite: (p) => p.replace(/^\/api/, '')  // 去掉 /api 前缀
      }
    }
  }
})
