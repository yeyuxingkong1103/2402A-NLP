// 第 9 步：前端路由表与登录守卫

import { createRouter, createWebHistory } from 'vue-router'   // 导入路由创建函数与 HTML5 历史模式

import Login from './views/Login.vue'                // 导入登录 / 注册页面
import User from './views/User.vue'                  // 导入用户页面
import RoleSelect from './views/RoleSelect.vue'      // 导入角色选择页面
import Chat from './views/Chat.vue'                  // 导入智能问答页面
import History from './views/History.vue'            // 导入历史记录页面
import KbManage from './views/KbManage.vue'          // 导入知识库管理页面
import UploadStatus from './views/UploadStatus.vue'  // 导入资料上传页面
import Admin from './views/Admin.vue'                // 导入后台管理页面
import Eval from './views/Eval.vue'                  // 导入评测结果页面

const routes = [                                     // 定义路由表：路径与页面组件一一对应
  { path: '/', redirect: '/login' },                 // 根路径重定向到登录页
  { path: '/login', component: Login },              // 登录页
  { path: '/user', component: User },                // 用户页
  { path: '/role', component: RoleSelect },          // 角色选择页
  { path: '/chat', component: Chat },                // 问答页
  { path: '/history', component: History },          // 历史记录页
  { path: '/kb', component: KbManage },              // 知识库管理页
  { path: '/upload', component: UploadStatus },      // 上传页
  { path: '/admin', component: Admin },              // 后台页
  { path: '/eval', component: Eval }                 // 评测页
]

const router = createRouter({                        // 创建路由实例
  history: createWebHistory(),                       // 使用 HTML5 历史模式，地址栏不带 #
  routes                                             // 挂载上面定义的路由表
})

router.beforeEach((to) => {                          // 全局前置守卫：未登录一律回登录页
  const userId = localStorage.getItem('user_id')     // 从本地存储读取用户编号
  if (to.path !== '/login' && !userId) {             // 目标不是登录页且没有用户编号
    return '/login'                                  // 中断当前跳转，改跳登录页
  }
  return true                                        // 其余情况放行
})

export default router                               // 导出路由实例，供 main.js 注册
