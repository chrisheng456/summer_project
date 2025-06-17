<template>
  <div id="app">
    <!-- 显示登录页 -->
    <LogInPage />

    <!-- 显示翻译页 -->
    <TranslatePage />
  </div>
</template>

<script>
import LogInPage from './components/LogInPage.vue'
import TranslatePage from './components/TranslatePage.vue'

export default {
  name: 'App',
  components: {
    LogInPage,
    TranslatePage
  },
  data() {
    return {
      loggedIn: false
    }
  }
}
</script>

<style>
/* 可选样式 */
</style>