<template>
  <div class="page-header">
    <!-- Left logo section -->
    <div class="header-title">
      <img src="@/assets/logo.png" alt="Logo" class="logo-image" />
    </div>

    <!-- Right user avatar dropdown -->
    <el-dropdown trigger="click" @command="handleCommand">
      <span class="avatar-wrapper">
        <el-avatar size="default">{{ userInitial }}</el-avatar>
      </span>

      <!-- Dropdown menu content -->
      <template #dropdown>
        <el-dropdown-menu>
          <!-- Show username (disabled, non-clickable) -->
          <el-dropdown-item disabled>{{ username }}</el-dropdown-item>
          
          <!-- Logout button -->
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

// On first load/refresh, load from local storage if available
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
  position: fixed;             
  top: 0;
  left: 0;
  right: 0;
  height: 40px;
  background-color: #ffffff !important;    
  display: flex;               
  justify-content: space-between; 
  align-items: center;          
  padding: 10px 24px;           
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

.logo-image {
  height: 80px;      
  object-fit: contain;
}
</style>