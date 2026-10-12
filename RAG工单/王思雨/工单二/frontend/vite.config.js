// 工单编号：人工智能NLP-RAG-基于PDF文档的问答系统优化
// Vite 配置：把 /api 请求代理到本机 FastAPI 服务，避免浏览器跨域问题。
import { defineConfig } from 'vite'                 // Vite 配置工厂
import vue from '@vitejs/plugin-vue'                // 支持 .vue 单文件组件

export default defineConfig({                       // 导出配置
  plugins: [vue()],                                 // 启用 Vue 插件
  server: {                                         // 开发服务器配置
    port: 5173,                                     // 前端端口
    proxy: { '/api': 'http://localhost:8000' }      // 后端接口代理目标
  }
})
