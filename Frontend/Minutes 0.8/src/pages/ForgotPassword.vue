<template>
  <div class="page-layout">
    <!-- 左边：图片+欢迎文字 -->
    <LeftImagePanel :imageUrl="loginBg" />

    <!-- 右边：找回密码框 -->
    <div class="page-wrapper">
      <div class="register-container">
        <!-- LOGO -->
        <img src="@/assets/logo.png" alt="AutoMinute Logo" class="logo" />
        <h2 class="register-title">Reset your password</h2>

        <input v-model="email" type="email" placeholder="Enter your email address" />

        <button class="register-button" @click="handleReset">Send Reset Link</button>

        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        <p v-if="successMessage" class="success">{{ successMessage }}</p>

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

const email = ref('')
const errorMessage = ref('')
const successMessage = ref('')

function goBack() {
  router.push('/LoginPage')
}

async function handleReset() {
  errorMessage.value = ''
  successMessage.value = ''

  if (!email.value) {
    errorMessage.value = 'Please enter your email address'
    return
  }

  try {
    const res = await axios.post('/api/forgot-password', {
      email: email.value
    })

    if (res.data.code === 200) {
      successMessage.value = 'Reset link sent to your email'
    } else {
      errorMessage.value = res.data.message || 'Failed to send reset link'
    }
  } catch (err) {
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

.success {
  color: green;
  margin-top: 10px;
}

.links {
  display: flex;
  justify-content: center;
  font-size: 14px;
}

a {
  color: black;
  text-decoration: none;
  margin: 0 10px;
}

.register-title {
  font-size: 25px;
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
