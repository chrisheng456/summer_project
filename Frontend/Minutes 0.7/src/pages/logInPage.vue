<template>
  <div class="page-layout">
    <!-- 左边：插画区 -->
    <LeftImagePanel :imageUrl="loginBg" />

    <!-- 右边：登录框 -->
    <div class="page-wrapper">
      <div class="login-container">
        <!-- LOGO -->
        <img src="@/assets/logo.png" alt="AutoMinute Logo" class="logo" />

        <!-- 欢迎语 -->
        <h2 class="welcome-title">Welcome! <br />Let's get started.</h2>

        <!-- 登录表单 -->
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
import loginBg from '@/assets/login-bg.jpg'

//  使用你封装好的模块（在 modules/index.ts 里 re-export 了 authApi）
import { authApi } from '@/api/modules/index'

// 用户信息管理
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

  // 简单校验
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
      // 后端未返回 user，这里用表单名兜底构造一个轻量用户对象
      const user = { id: 0, name: username.value }

      // Pinia：全局保存
      auth.login(user, res.token)

      // localStorage：给 axios 拦截器自动带 Authorization
      localStorage.setItem('token', res.token)

      // 缓存会议（可选）
      if (res.meetings) {
        localStorage.setItem('meetings', JSON.stringify(res.meetings))
      }

      router.push('/UploadHistory')
    } else {
      // 后端自定义 message 时兜底
      errorMessage.value = 'Login failed'
    }
  } catch (e: any) {
    // 兜底错误信息
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
/* 你的样式保持不变，只补一个禁用态友好点 */
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