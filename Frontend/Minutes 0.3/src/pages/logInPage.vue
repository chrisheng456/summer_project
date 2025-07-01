<template>
  <div class="page-wrapper">
    <div class="login-container">
      <h2>LOGO</h2>
      <input v-model="username" type="text" placeholder="Username" />
      <input v-model="password" type="password" placeholder="Password" />
      <button class="login-button" @click="handleLogin">Login</button>
      <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
      <div class="links">
        <a href="#">Forgot Password</a>
        <a href="#">Register Account</a>
      </div>
    </div>
  </div>
</template>

<script lang="ts" setup name="LoginPage">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

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
      console.log('Login success')
      router.push('/UploadHistory') // 登录成功跳转页面
    } else {
      errorMessage.value = res.data.message || 'Login failed'
    }
  } catch (error) {
    errorMessage.value = 'Network or server error'
    console.error('Login error:', error)
  }
}
</script>

<style scoped>
.page-wrapper {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: url('background.jpg') no-repeat center center / cover;
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
  background-color: #14191f;
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
  color: white;
  text-decoration: none;
}
</style>