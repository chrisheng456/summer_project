<template>
  <div class="page-wrapper">
    <div class="login-container">
      <h2>LOGO</h2>
      <input v-model="username" type="text" placeholder="用户名" />
      <input v-model="password" type="password" placeholder="密码" />
      <button class="login-button" @click="handleLogin">登录</button>
      <div class="links">
        <a href="#">忘记密码</a>
        <a href="#">注册账号</a>
        <a href="#">其他帮助</a>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup name =logInPage>
import { ref } from 'vue'
import { RouterView,useRouter} from 'vue-router';

const router = useRouter()
const username = ref('')
const password = ref('')

function handleLogin() {
  router.push('/translatePage') 
  console.log('登录中：', username.value, password.value)
  // 可以用 axios 发送请求
  
}
</script>

<style scoped>
.page-wrapper {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: url('background.jpg') no-repeat center center/cover;
}

.login-container {
  background-color: rgba(0, 0, 0, 0.5);
  padding: 40px;
  border-radius: 10px;
  text-align: center;
  width: 300px;
  color: white;
  font-family: Arial, sans-serif;
}

input {
  width: 80%;
  padding: 10px;
  margin: 10px 0;
  border: none;
  border-radius: 5px;
}

.login-button {
  width: 80%;
  padding: 10px;
  margin: 20px 0;
  background-color: #007BFF;
  border: none;
  border-radius: 5px;
  color: white;
  cursor: pointer;
}

.links {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
}

a {
  color: white;
  text-decoration: none;
}
</style>