// 第 9 步：Vite 构建配置，增加 /api 代理、固定端口与路径别名

import { defineConfig } from 'vite'          // 导入 Vite 的配置函数
import vue from '@vitejs/plugin-vue'         // 导入 Vue 官方插件
import { fileURLToPath, URL } from 'node:url'   // 导入路径转换工具，用于解析别名绝对路径

export default defineConfig({                // 导出 Vite 配置对象
  plugins: [vue()],                          // 启用 Vue 单文件组件支持
  server: {                                  // 开发服务器配置
    port: 5173,                              // 前端开发服务器端口，固定为 5173
    open: false,                             // 启动后不自动打开浏览器
    proxy: {                                 // 反向代理配置：解决开发期跨域
      '/api': {                              // 所有以 /api 开头的请求
        target: 'http://127.0.0.1:8000',     // 转发到本地后端服务
        changeOrigin: true,                  // 改写请求头 Host，避免后端校验失败
        rewrite: (path) => path.replace(/^\/api/, '')   // 去掉 /api 前缀，后端路由不带它
      }
    }
  },
  resolve: {                                 // 模块解析配置
    alias: {                                 // 路径别名配置
      '@': fileURLToPath(new URL('./src', import.meta.url))   // 用 @ 指向 src 目录的绝对路径
    }
  }
})
