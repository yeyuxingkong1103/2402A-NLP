// 第 9 步：前端入口文件，注册路由与 Element Plus 组件库

import { createApp } from 'vue'                                  // 导入创建 Vue 应用的函数
import ElementPlus from 'element-plus'                           // 导入 Element Plus 组件库主体
import 'element-plus/dist/index.css'                             // 导入 Element Plus 全局样式
import * as ElementPlusIconsVue from '@element-plus/icons-vue'   // 导入全部 Element Plus 图标

import App from './App.vue'                                      // 导入根组件
import router from './router'                                    // 导入路由实例（含登录守卫）

const app = createApp(App)                                       // 创建应用实例，根组件为 App

for (const [name, component] of Object.entries(ElementPlusIconsVue)) {   // 遍历全部图标组件
  app.component(name, component)                                 // 逐个全局注册，模板里可直接用 <el-icon><Search /></el-icon>
}

app.use(router)                                                  // 注册路由，页面切换生效
app.use(ElementPlus)                                             // 注册 Element Plus，全站可用 el-* 组件
app.mount('#app')                                                // 挂载到 index.html 里的 #app 容器
