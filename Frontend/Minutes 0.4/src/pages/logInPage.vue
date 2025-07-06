<template>
  <div class="page-layout">
    <!-- 左边：图片+欢迎文字 -->
    <LeftImagePanel imageUrl="/assets/login-bg.jpg">
      <div class="overlay-content">
        <p style="color: #eee">AI meeting assistant</p>
      </div>
    </LeftImagePanel>

    <!-- 右边：登录框 -->
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
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import LeftImagePanel from '@/components/LeftImagePanel.vue'
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

      console.log("成功登录")
      router.push('/UploadHistory')
    } else {
      errorMessage.value = res.data.message || 'Login failed'
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

.login-container {
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
  color: white;
  text-decoration: none;
}

</style>

