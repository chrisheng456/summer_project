<template>
  <div>
    <!-- 页面顶部栏 -->
    <HeaderBar />

    <!-- 内容区加外层容器 -->
    <div class="container">

      <!-- 上传音频区域 -->
      <div class="upload-section">
        <h3>Upload New Audio</h3>

        <!-- Element Plus 上传组件：支持拖拽、自定义处理上传功能 -->
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

      <!-- 上传历史表格区域 -->
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
  </div>
</template>

<script lang="ts" setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import HeaderBar from '@/components/HeaderBar.vue'

// 定义历史数据类型
interface HistoryItem {
  id: string
  filename: string
  participants: number
  duration: string
  uploadTime: string
}

const router = useRouter()
const history = ref<HistoryItem[]>([])

// 用户名和头像首字母
const username = 'Gaoxinjie'
const userInitial = username[0].toUpperCase()

// 处理用户操作
function handleCommand(command: string) {
  if (command === 'logout') {
    ElMessage.success('Logged out')
    router.push('/login')
  } else if (command === 'settings') {
    ElMessage.info('Settings page coming soon')
  }
}

// 处理上传文件（模拟方式）
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

// 跳转到详情页
function viewDetail() {
  router.push(`/MeetingNotes`)
}
</script>

<style scoped>
/* 页面主容器：用于整体布局和背景设置 */
.container {
  display: flex;                       /* 垂直排列 */
  flex-direction: column;
  align-items: center;                  /* 居中对齐 */
  padding: 40px 20px;                   /* 上下左右内边距 */
  min-height: 100vh;                    /* 最小高度铺满屏幕 */
  background-color: #f5f7fa;            /* 浅灰背景 */
  box-sizing: border-box;
  font-family: 'Helvetica Neue', Arial, sans-serif;  /* 设置字体 */
}

/* 上传区域：卡片样式 */
.upload-section {
  background-color: #fff;               /* 白色背景 */
  padding: 20px 20px;
  border-radius: 12px;                  /* 圆角 */
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.337);  /* 阴影 */
  width: 100%;
  max-width: 1000px;
  margin-bottom: 40px;                  /* 与下方内容间距 */
  text-align: center;
}

/* 上传区域：鼠标悬停效果 */
.upload-section:hover {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);  /* 增强阴影 */
  transform: translateY(-4px);                /* 上移产生悬浮感 */
  transition: all 0.3s ease;
}

/* 上传区域标题样式 */
.upload-section h3 {
  margin-bottom: 16px;
  font-size: 1.5rem;
  color: #333;
  font-weight: 600;
}

/* 表格外层容器：用于显示上传历史 */
.table-wrapper {
  background-color: #fff;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.337);
  width: 100%;
  max-width: 1000px;
  height: 320px;                        /* 固定高度 */
  overflow-y: auto;                     /* 超出滚动 */
}

/* 表格标题样式 */
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

/* 表格本体 */
table {
  width: 100%;
  border-collapse: collapse;            /* 去除间隙 */
  table-layout: fixed;                  /* 固定列宽 */
  font-size: 15px;
}

/* 表格表头样式 */
thead th {
  position: sticky;                     /* 表头固定 */
  top: 0;
  background-color: #f4f6f9;
  font-weight: 700;
  padding: 14px 12px;
  text-align: left;
  border-bottom: 2px solid #ddd;
  color: #333;
}

/* 表格单元格样式 */
tbody td {
  padding: 14px 12px;
  border-bottom: 1px solid #eee;
  word-break: break-word;               /* 单词换行 */
  color: #555;
}

/* 表格行悬停效果 */
tbody tr {
  transition: all 0.3s ease;
}

tbody tr:hover {
  background-color: #f0f8ff;            /* 淡蓝悬停色 */
  transform: translateY(-3px);          /* 微上移 */
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

/* 按钮通用样式 */
button {
  padding: 8px 16px;
  background-color: #409eff;            /* 蓝色按钮 */
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 14px;
  font-weight: 500;
  box-shadow: 0 2px 6px rgba(64, 158, 255, 0.4);
}

/* 按钮悬停效果 */
button:hover {
  background-color: #66b1ff;
  transform: translateY(-1px);
}

/* 移动端优化：小屏幕适配 */
@media (max-width: 768px) {
  .table-wrapper {
    max-width: 100%;
    overflow-x: auto;                   /* 横向滚动以适应表格 */
  }
  table {
    font-size: 13px;                    /* 缩小字体 */
  }
  .upload-section {
    padding: 16px;
  }
  .upload-section h3 {
    font-size: 1.2rem;
  }
}

/* 顶部导航栏Header固定样式 */
.page-header {
  position: fixed;                      /* 固定定位 */
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
  z-index: 10000;                       /* 保证在最上层 */
}

/* Header标题样式 */
.header-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

/* 头像鼠标效果 */
.avatar-wrapper {
  cursor: pointer;
}
</style>