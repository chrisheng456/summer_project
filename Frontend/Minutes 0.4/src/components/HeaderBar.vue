<template>
  <div class="page-header">
    <!-- 左侧Logo区域 -->
    <div class="header-title">Logo</div>

    <!-- 右侧用户头像下拉菜单 -->
    <el-dropdown trigger="click" @command="handleCommand">
      <span class="avatar-wrapper">
        <el-avatar size="medium">{{ userInitial }}</el-avatar>
      </span>

      <!-- 下拉菜单内容 -->
      <template #dropdown>
        <el-dropdown-menu>
          <!-- 显示用户名，不可点击 -->
          <el-dropdown-item disabled>{{ username }}</el-dropdown-item>
          
          <!-- 设置按钮 -->
          <el-dropdown-item divided command="settings">Settings</el-dropdown-item>
          
          <!-- 退出按钮 -->
          <el-dropdown-item command="logout">Log out</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script lang="ts" setup>
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

// 创建路由对象
const router = useRouter()

// 用户名（可改为从登录信息获取）
const username = 'Gaoxinjie'

// 获取用户名首字母大写，用作头像
const userInitial = username[0].toUpperCase()

// 处理下拉菜单点击事件
function handleCommand(command: string) {
  if (command === 'logout') {
    // 点击退出，提示消息并跳转到登录页
    ElMessage.success('Logged out')
    router.push('/login')
  } else if (command === 'settings') {
    // 点击设置，弹出提示
    ElMessage.info('Settings page coming soon')
  }
}
</script>

<style scoped>
.page-header {
  position: fixed;              /* 固定在页面顶部 */
  top: 0;
  left: 0;
  right: 0;
  height: 40px;
  background-color: #f9fafb;    /* 浅色背景 */
  display: flex;                /* 横向布局 */
  justify-content: space-between; /* 两端对齐 */
  align-items: center;          /* 垂直居中 */
  padding: 10px 24px;           /* 上下左右内边距 */
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.1); /* 阴影 */
  z-index: 10000;               /* 保证顶层显示 */
}

.header-title {
  font-size: 18px;              /* 字体大小 */
  font-weight: 600;             /* 字体加粗 */
  color: #333;                  /* 字体颜色 */
}

.avatar-wrapper {
  cursor: pointer;              /* 鼠标悬停变手型 */
}
</style>