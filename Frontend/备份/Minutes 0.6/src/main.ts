//引入createApp用于创建应用
import {createApp} from 'vue'
//引入App根组件
import App from './App.vue'
//引入路由器
import router from './router'
// 如果你使用 Element Plus：
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
//引入pinia
import { createPinia } from 'pinia'
// 只在开发环境加载 mock 文件
if (import.meta.env.MODE === 'development') {
  import('./shims/mock'),
  import ('./shims/mockLogin')
}

import axios from 'axios'

//创建一个app
const app =createApp(App)
const pinia =createPinia();

app.use(router)// app安装插件：使用路由和 Element Plus（或其他组件库）
app.use(ElementPlus)
app.use(pinia); // 直接传入 createPinia() 的结果
app.mount('#app')//挂载整个app到id=app中

// 设置基础URL（指向后端）
axios.defaults.baseURL = 'http://localhost:3000'

// 请求拦截器（自动添加token）
axios.interceptors.request.use(config => {
  const token = localStorage.getItem('token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export default axios