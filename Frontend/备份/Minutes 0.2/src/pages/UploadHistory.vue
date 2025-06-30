<template>
  <div class="container">
    <!-- 上传区域 -->
    <div class="upload-section">
      <h3>上传新音频</h3>
      <input type="file" @change="handleUpload" />
    </div>

    <!-- 历史记录 -->
    <div class="history-section">
      <h3 class="table-title">历史记录</h3>
      <div class="table-wrapper">
        <table>
          <thead >
            <tr>
              <th>文件名</th>
              <th>参众人数</th>
              <th>会议时长</th>
              <th>上传时间</th>
              <th>详情</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in history" :key="item.id">
              <td>{{ item.filename }}</td>
              <td>{{ item.participants }}</td>
              <td>{{ item.duration }}</td>
              <td>{{ item.uploadTime }}</td>
              <td>
                <button @click="viewDetail">View Detail</button>
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

interface HistoryItem {
  id: string
  filename: string
  participants: number
  duration: string
  uploadTime: string
}

const router = useRouter()
const history = ref<HistoryItem[]>([])

function handleUpload(event: Event) {
  const fileInput = event.target as HTMLInputElement
  const file = fileInput.files?.[0]
  if (file) {
    const newItem: HistoryItem = {
      id: Date.now().toString(),
      filename: file.name,
      participants: Math.floor(Math.random() * 10 + 1),
      duration: `${Math.floor(Math.random() * 60)} min`,
      uploadTime: new Date().toLocaleString()
    }
    history.value.unshift(newItem)
  }
}

function viewDetail() {
  router.push(`/MeetingSummary`)
}
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 40px 20px;
  min-height: 100vh;
  background-color: #f5f7fa;
  box-sizing: border-box;
}

/* 上传区域样式 */
.upload-section {
  margin-bottom: 40px;
  text-align: center;
  background-color: #ffffff;
  padding: 20px 30px;
  border-radius: 8px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
  width: 100%;
  max-width: 600px;
}

.upload-section h3 {
  margin-bottom: 16px;
}

/* 历史记录表格区域 */
.table-wrapper {
  width: 100%;
  max-width: 1000px;
  height: 320px;
  overflow-y: auto;
  border-radius: 8px;
  background: #ffffff;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
  padding: 16px;
}

.table-title {
  text-align: center;
  font-size: 20px;
  font-weight: bold;
  margin-bottom: 12px;
  color: #333;
}

table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
}

thead th {
  position: sticky;
  top: 0;
  background-color: #e8ebf0;
  font-weight: bold;
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #ddd;
}

tbody td {
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
  word-break: break-word;
}

/* 按钮样式 */
button {
  padding: 6px 12px;
  background-color: #409eff;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

button:hover {
  background-color: #66b1ff;
}
</style>
