<template>
  <div class="page-layout">
    <!-- 左边：图片+欢迎文字 -->
    <LeftImagePanel :imageUrl="loginBg" />

    <!-- 右边：注册框 -->
    <div class="page-wrapper">
      <div class="register-container">
        <h2>Register</h2>
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
  if (password.value !== confirmPassword.value) {
    errorMessage.value = 'Passwords do not match'
    return
  }
  try {
    const res = await axios.post('/api/register', {
      username: username.value,
      email: email.value,
      password: password.value,
    })

    if (res.data.code === 200) {
      // 注册成功后跳转登录页面或其他逻辑
      router.push('/login')
    } else {
      errorMessage.value = res.data.message || 'Registration failed'
    }
  } catch (e) {
    errorMessage.value = 'Network error'
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
  background-color: rgba(230, 231, 231, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
}

.register-container {
  background-color: rgba(182, 184, 184, 0.457);
  padding: 40px;
  border-radius: 10px;
  text-align: center;
  width: 300px;
  color: rgba(37, 28, 28, 0.741);
}

input {
  width: 80%;
  padding: 10px;
  margin: 10px 0;
  border: none;
  border-radius: 5px;
  box-sizing: border-box;
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
  color: white;
  text-decoration: none;
  margin: 0 10px;
}
</style>
