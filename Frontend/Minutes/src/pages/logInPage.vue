<template>
  <div class="page-layout">
    <!-- Left side: illustration panel -->
    <LeftImagePanel :imageUrl="loginBg" />

    <!-- Right side: login form -->
    <div class="page-wrapper">
      <div class="login-container">
        <!-- LOGO -->
        <img src="@/assets/logo.png" alt="AutoMinute Logo" class="logo" />

        <!-- Welcome text -->
        <h2 class="welcome-title">Welcome! <br />Let's get started.</h2>

        <!-- Login form inputs -->
        <input
          v-model.trim="username"
          type="text"
          placeholder="Username"
          @keyup.enter="handleLogin"
        />
        <input
          v-model="password"
          type="password"
          placeholder="Password"
          @keyup.enter="handleLogin"
        />

        <button class="login-button" :disabled="loading" @click="handleLogin">
          {{ loading ? 'Logging in...' : 'Login' }}
        </button>

        <!-- Error message -->
        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>

        <!-- Links -->
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
import loginBg from '@/assets/login-bg.jpg'
import { authApi } from '@/api/modules/index'

import { useAuthStore } from '@/stores/user'
const auth = useAuthStore()

const router = useRouter()
const username = ref('')
const password = ref('')
const errorMessage = ref('')
const loading = ref(false)

async function handleLogin() {
  if (loading.value) return
  errorMessage.value = ''

  // Basic validation
  if (!username.value || !password.value) {
    errorMessage.value = 'Please enter username and password'
    return
  }
  try {
    loading.value = true
    const res = await authApi.login({
      username: username.value,
      password: password.value,
    })

      if (res.ok && res.token) {
      const user = { id: 0, name: username.value }
      auth.login(user, res.token)

      localStorage.setItem('token', res.token)
      if (res.meetings) {
        localStorage.setItem('meetings', JSON.stringify(res.meetings))
      }

      router.push('/UploadHistory')
    } else {
      errorMessage.value = 'Login failed'
    }
  } catch (e: any) {
    // Error handling
    errorMessage.value =
      e?.response?.data?.message ||
      e?.response?.data?.detail ||
      'Network error'
  } finally {
    loading.value = false
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
  flex-direction: row;
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
.login-button[disabled] {
  opacity: 0.7;
  cursor: not-allowed;
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
@media (max-width: 768px) {
  .page-layout {
    flex-direction: column;
  }
  .page-wrapper {
    align-items: center;
    padding-top: 20px;
  }
}
</style>