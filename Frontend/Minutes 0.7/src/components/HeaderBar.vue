<template>
  <div class="page-header">
    <!-- 左侧Logo区域 -->
    <div class="header-title">
      <img src="@/assets/logo.png" alt="Logo" class="logo-image" />
    </div>

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
          
          <!-- 退出按钮 -->
          <el-dropdown-item command="logout">Log out</el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </div>
</template>


<script lang="ts" setup>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { useAuthStore } from '@/stores/user'

const router = useRouter()
const auth = useAuthStore()

// 首次进入/刷新，从本地载入一次
onMounted(() => {
  if (!auth.user && !auth.token) auth.loadFromStorage()
})

const username = computed(() => auth.user?.name || 'Guest')
const userInitial = computed(() => username.value.charAt(0).toUpperCase())

function handleCommand(command: string) {
  if (command === 'logout') {
    auth.logout()
    ElMessage.success('Logged out')
    router.push('/loginPage')
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
  background-color: #ffffff !important;    
  display: flex;                /* 横向布局 */
  justify-content: space-between; /* 两端对齐 */
  align-items: center;          /* 垂直居中 */
  padding: 10px 24px;           /* 上下左右内边距 */
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

.logo-image {
  height: 80px;       /* 可根据需要调整大小 */
  object-fit: contain;
}
</style>