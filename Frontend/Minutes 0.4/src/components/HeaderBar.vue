<template>
  <div class="page-header">
    <div class="header-title">Meeting Dashboard</div>
    <el-dropdown trigger="click" @command="handleCommand">
      <span class="avatar-wrapper">
        <el-avatar size="medium">{{ userInitial }}</el-avatar>
      </span>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item disabled>{{ username }}</el-dropdown-item>
          <el-dropdown-item divided command="settings">Settings</el-dropdown-item>
          <el-dropdown-item command="logout">Log out</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>

<script lang="ts" setup>
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const router = useRouter()
const username = 'Gaoxinjie'
const userInitial = username[0].toUpperCase()

function handleCommand(command: string) {
  if (command === 'logout') {
    ElMessage.success('Logged out')
    router.push('/login')
  } else if (command === 'settings') {
    ElMessage.info('Settings page coming soon')
  }
}
</script>

<style scoped>
.page-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 60px;
  background-color: #f9fafb;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.1);
  z-index: 10000;
}

.header-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.avatar-wrapper {
  cursor: pointer;
}
</style>
