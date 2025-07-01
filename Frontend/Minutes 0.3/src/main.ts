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
  import ('./mock/mockLogin')
}

//创建一个app
const app =createApp(App)
const pinia =createPinia();

app.use(router)// app安装插件：使用路由和 Element Plus（或其他组件库）
app.use(ElementPlus)
app.use(pinia); // 直接传入 createPinia() 的结果
app.mount('#app')//挂载整个app到id=app中