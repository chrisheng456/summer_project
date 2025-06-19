//引入createApp用于创建应用
import {createApp} from 'vue'
//引入App根组件
import App from './App.vue'
//引入路由器
import router from './router'
//创建一个app
const app =createApp(App)
//引入路由器
app.use(router)
//挂载整个app到id=app中
app.mount('#app')