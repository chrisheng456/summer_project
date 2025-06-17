<template>
  <div class="login-container">
    <div>
      <h2>LOGO</h2>
    </div>
    <input v-model="username" type="text" placeholder="用户名" />
    <input v-model="password" type="password" placeholder="密码" />
    <button class="login-button" @click="handleLogin">登录</button>
    <div class="links">
      <a href="#">忘记密码</a>
      <a href="#">注册账号</a>
      <a href="#">其他帮助</a>
    </div>
  </div>
</template>

<script>
export default {
  name: "LoginPage",
  data() {
    return {
      username: "",
      password: ""
    };
  },
  methods: {
    handleLogin() {
      // 这里可以调用接口，例如 axios.post('/api/login', { username, password })
      console.log("登录中：", this.username, this.password);
    }
  }
};
</script>

<style scoped>
body {
  margin: 0;
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  font-family: Arial, sans-serif;
  color: white;
}
.login-container {
  background-color: rgba(0, 0, 0, 0.5);
  padding: 40px;
  border-radius: 10px;
  text-align: center;
  width: 300px;
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