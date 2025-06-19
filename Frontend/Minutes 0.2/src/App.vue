<template>
  <div id="app">
    <!-- 用 v-if 控制是否显示欢迎内容 -->
    <div class="main-content" v-if="showMain">
      <h1>欢迎</h1>
      <div @click="start" class="start-button">点击开始使用</div>

    </div>
    <!-- RouterView 永远展示 -->
    <RouterView></RouterView>

  </div>
</template>

<script lang ="ts" setup name ="App">//vue3写法，组件名写在<script>里面
// import LogInPage from './components/logInPage.vue'//直接倒入不用写component{}注册，因为vue3 import自动return
// import TranslatePage from './components/translatePage.vue'
import { ref } from 'vue'
import { RouterView,useRouter} from 'vue-router';

const router = useRouter()
const showMain = ref(true)  // 控制欢迎页面是否显示

function start(){
    showMain.value = false      // 点击后隐藏 main-content
    router.push('/logInPage')   // 跳转到登录页

}
</script> 

<!-- 加入局部样式，使得css作用域是本vue的html -->
<style scoped>
.main-content {
  text-align: center;
  margin-top: 100px;
/* 可选样式 */}

.start-button {
  border: 2px solid black; /* 设置边框 */
  padding: 10px 20px;      /* 内边距 */
  display: inline-block;   /* 保持块级但适配内容 */
  cursor: pointer;         /* 鼠标变成小手 */
  border-radius: 5px;      /* 可选：圆角 */
}
</style>