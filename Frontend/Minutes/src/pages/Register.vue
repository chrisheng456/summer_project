<template>
  <div class="page-layout">
    <!-- Left: image + welcome text -->
    <LeftImagePanel :imageUrl="loginBg" />

    <!-- Right: register form -->
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

// Form data
const username = ref('')
const email = ref('')
const password = ref('')
const confirmPassword = ref('')
const errorMessage = ref('')

// Navigate back to login page
function goBack() {
  router.push('/LoginPage')
}

// Register logic
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
