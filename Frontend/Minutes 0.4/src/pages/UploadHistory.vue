<template>
  <div class="container">
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

    <!-- Upload Section -->
    <div class="upload-section">
      <h3>Upload New Audio</h3>
      <el-upload
        class="upload-card"
        action="#"
        drag
        :auto-upload="false"
        :on-change="handleUpload"
        accept=".mp3,.wav,.m4a"
        :limit="100"
        :file-list="[]"
      >
        <el-icon class="upload-icon">
          <UploadFilled />
        </el-icon>

        <div class="el-upload__text">
          Drop audio file here or <em>click to upload</em>
        </div>
        
        <template #tip>
          <div class="el-upload__tip">Only mp3/wav/m4a files</div>
        </template>
      </el-upload>

    </div>

    <!-- History Table -->
    <div class="history-section">
      <h3 class="table-title">Upload History</h3>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Participants</th>
              <th>Duration</th>
              <th>Uploaded At</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in history" :key="item.id">
              <td>{{ item.filename }}</td>
              <td>{{ item.participants }}</td>
              <td>{{ item.duration }}</td>
              <td>{{ item.uploadTime }}</td>
              <td>
                <button @click="viewDetail">View Details</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>



<script lang="ts" setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'


interface HistoryItem {
  id: string
  filename: string
  participants: number
  duration: string
  uploadTime: string
}

const router = useRouter()
const history = ref<HistoryItem[]>([])


const username = 'Gaoxinjie'
const userInitial = username[0].toUpperCase()

function handleCommand(command: string) {
  if (command === 'logout') {
    ElMessage.success('Logged out')
    router.push('/login')  // 回到登录页
  } else if (command === 'settings') {
    ElMessage.info('Settings page coming soon')
  }
}

function handleUpload(file: any) {
  const rawFile = file.raw
  if (rawFile) {
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      filename: rawFile.name,
      participants: Math.floor(Math.random() * 10 + 1),
      duration: `${Math.floor(Math.random() * 60)} min`,
      uploadTime: new Date().toLocaleString()
    }
    history.value.unshift(newItem)
    ElMessage.success('Upload success (mocked)')
  }
}

function viewDetail() {
  router.push(`/MeetingNotes`)
}
</script>

<style scoped>
/* 页面容器 */
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px 20px;
  min-height: 100vh;
  background-color: #f5f7fa;
  box-sizing: border-box;
  font-family: 'Helvetica Neue', Arial, sans-serif;
}

/* 上传区域 */
.upload-section {
  background-color: #fff;
  padding: 20px 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.337);
  width: 100%;
  max-width: 1000px;
  margin-bottom: 40px;
  text-align: center;
}

.upload-section:hover {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2); /* 增加阴影，让卡片悬浮感更强 */
  transform: translateY(-4px);                /* 轻微上移 */
  transition: all 0.3s ease;                  /* 添加过渡动画，提升流畅度 */
}



.upload-section h3 {
  margin-bottom: 16px;     /* 标题底部留 16 像素的间距（和上传框之间有空隙） */
  font-size: 1.5rem;       /* 字体大小为 1.5 倍的根字体大小，通常是 24px（如果根字体是 16px） */
  color: #333;             /* 字体颜色为深灰色（不是纯黑，更柔和） */
  font-weight: 600;        /* 字体加粗程度为 600（比 normal 粗、比 bold 轻） */
}


input[type="file"] {
  border: 2px dashed #409eff;
  padding: 16px;
  border-radius: 10px;
  background-color: #f8fbff;
  cursor: pointer;
  transition: border-color 0.3s ease;
  width: 100%;
  max-width: 500px;
}

input[type="file"]:hover {
  border-color: #66b1ff;
}

/* 表格外层容器 */
.table-wrapper {
  background-color: #fff;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.337);
  width: 100%;
  max-width: 1000px;
  height: 320px;
  overflow-y: auto;
}


/* 表格标题 */
.table-title {
  text-align: center;
  font-size: 1.5rem;
  font-weight: 700;
  margin: 16px auto 20px;
  padding: 10px 24px;
  color: #333;
  border-radius: 8px;
  border: 1px solid #d0d4d9;
  width: fit-content;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
}

/* 表格结构 */
table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 15px;
}

thead th {
  position: sticky;
  top: 0;
  background-color: #f4f6f9;
  font-weight: 700;
  padding: 14px 12px;
  text-align: left;
  border-bottom: 2px solid #ddd;
  color: #333;
}

tbody td {
  padding: 14px 12px;
  border-bottom: 1px solid #eee;
  word-break: break-word;
  color: #555;
}

tbody tr {
  transition: all 0.3s ease;
}

tbody tr:hover {
  background-color: #f0f8ff; /* 淡蓝背景可选 */
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}


/* 按钮样式 */
button {
  padding: 8px 16px;
  background-color: #409eff;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
  font-weight: 500;
  box-shadow: 0 2px 6px rgba(64, 158, 255, 0.4);
}

button:hover {
  background-color: #66b1ff;
  transform: translateY(-1px);
}

/* 移动端响应式优化 */
@media (max-width: 768px) {
  .table-wrapper {
    max-width: 100%;
    overflow-x: auto;
  }

  table {
    font-size: 13px;
  }

  .upload-section {
    padding: 16px;
  }

  .upload-section h3 {
    font-size: 1.2rem;
  }
}

/* 固定头像在右上角 */
.avatar-wrapper {
  position: fixed;
  top: 20px;
  right: 30px;
  z-index: 9999;
  cursor: pointer;
}



</style>
