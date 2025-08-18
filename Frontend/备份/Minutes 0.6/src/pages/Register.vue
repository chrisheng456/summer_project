<template>
  <div class="page-layout">
    <!-- 左边：图片+欢迎文字 -->
    <LeftImagePanel :imageUrl="loginBg" />

    <!-- 右边：注册框 -->
    <div class="page-wrapper">
      <div class="register-container">
        <!-- LOGO -->
        <img src="@/assets/logo.png" alt="AutoMinute Logo" class="logo" />   
        <h2 class="register-title">Create your account</h2>
        <input v-model="username" type="text" placeholder="Username" />
        <input v-model="email" type="email" placeholder="Email" />
        <input v-model="password" type="password" placeholder="Password" />
        <input v-model="confirmPassword" type="password" placeholder="Confirm Password" />
        <button class="register-button" @click="handleRegister">Register</button>
        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        <div class="links">
          <a href="#" @click.prevent="goBack">Back to Login</a>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import LeftImagePanel from '@/components/LeftImagePanel.vue'
import loginBg from '@/assets/login-bg.jpg'
import axios from 'axios'

const router = useRouter()

// 表单数据
const username = ref('')
const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const errorMessage = ref('')

// 返回登录页面
function goBack() {
  router.push('/LoginPage')
}

// 注册逻辑
async function handleRegister() {
  // 密码确认检查
  if (password.value !== confirmPassword.value) {
    errorMessage.value = 'Passwords do not match'
    return
  }

  try {
    await axios.post('/api/register', {
      username: username.value,
      email: email.value,
      password: password.value
    })
    
    // 注册成功后跳转登录页
    router.push('/LoginPage')
  } catch (e: any) {
    errorMessage.value = e.response?.data?.error || 
                        e.message || 
                        'Registration failed'
  }
}

</script>

<style scoped>
.page-layout {
  display: flex;
  height: 100vh;
}

.page-wrapper {
  flex: 1;  
  background-color: white;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding-top: 50px;
}

.register-container {
  background-color: white;
  padding: 40px;
  border-radius: 10px;
  text-align: center;
  width: 300px;
  color: rgba(37, 28, 28, 0.741);
}

input {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
  border: 1px solid black;
  border-radius: 5px;
}

.register-button {
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
  justify-content: center;
  font-size: 14px;
}

a {
  color: balck;
  text-decoration: none;
  margin: 0 10px;
}

.register-title {
  font-size: 30px;
  font-weight: bold;
  margin-bottom: 15px;
  color: #333;
  text-align: left;
}

.logo {
  height: 80px;
  margin-bottom: 20px;
}
</style>
