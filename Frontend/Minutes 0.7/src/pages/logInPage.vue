<template>
  <div class="page-layout">
    <!-- 左边：插画区 -->
    <LeftImagePanel :imageUrl="loginBg">
    </LeftImagePanel>

    <!-- 右边：登录框 -->
    <div class="page-wrapper">
      <div class="login-container">
        <!-- LOGO -->
        <img src="@/assets/logo.png" alt="AutoMinute Logo" class="logo" />   
        <!-- 欢迎语 -->
        <h2 class="welcome-title">Welcome! <br />Let's get started.</h2>
        <!-- 登录表单 -->
        <input v-model="username" type="text" placeholder="Username" />
        <input v-model="password" type="password" placeholder="Password" />
        <button class="login-button" @click="handleLogin">Login</button>
        <!-- 错误信息 -->
        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        <!-- 其他链接 -->
        <div class="links">
          <a @click.prevent="goToForgotPassword" style="cursor: pointer;">Forgot Password</a>
          <a @click.prevent="goToRegister" style="cursor: pointer;">Register Account</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import LeftImagePanel from '@/components/LeftImagePanel.vue'
import axios from 'axios'
import loginBg from '@/assets/login-bg.jpg'

const router = useRouter()
const username = ref('')
const password = ref('')
const errorMessage = ref('')

async function handleLogin() {
  try {
    const res = await axios.post('/api/login', {
      username: username.value,
      password: password.value,
    })

    if (res.data.code === 200) {

      console.log("Login successful")
      router.push('/UploadHistory')
    } else {
      errorMessage.value = res.data.message || 'Login failed'
    }
  } catch (e) {
    errorMessage.value = 'Network error'
  }
}
function goToRegister() {
  router.push('/Register')
}

function goToForgotPassword() {
  router.push('/ForgotPassword')
}
</script>

<style scoped>

.page-layout {
  display: flex;
  flex-direction: row; /* 默认方向为行 */
  height: 100vh;
  background-color: white;
}

.page-wrapper {
  flex: 1;  
  background-color: white;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 50px;
}

.login-container {
  background-color: white;
  padding: 40px;
  border-radius: 10px;
  text-align: center;
  width: 350px;
  color: rgba(37, 28, 28, 0.741);

}

.logo {
  height: 80px;
  margin-bottom: 20px;
}

.welcome-title {
  font-size: 30px;
  font-weight: bold;
  margin-bottom: 15px;
  color: #333;
  text-align: left;
}

input {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
  border: 1px solid black;
  border-radius: 5px;
}

.login-button {
  width: 80%;
  padding: 10px;
  margin: 20px 0;
  background-color: #3c1cf1af;
  border: none;
  border-radius: 5px;
  color: white;
  cursor: pointer;
}

.error {
  color: red;
  margin-top: 10px;
}

.links {
  display: flex;
  justify-content: space-between;
  font-size: 14px;
}

a {
  color: black;
  text-decoration: none;
}

/* 响应式处理 */
@media (max-width: 768px) {
  .page-layout {
    flex-direction: column; /* 小屏幕时改为列方向 */
  }

  .page-wrapper {
    align-items: center;
    padding-top: 20px;
  }
}
</style>

