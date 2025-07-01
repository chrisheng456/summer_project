<template>
  <div class="container">
    <!-- Upload Section -->
    <div class="upload-section">
      <h3>Upload New Audio</h3>
      <input type="file" @change="handleUpload" />
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
  padding: 24px 32px;
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
  width: 100%;
  max-width: 600px;
  margin-bottom: 40px;
  text-align: center;
}

.upload-section h3 {
  margin-bottom: 16px;
  font-size: 1.25rem;
  color: #333;
}

/* 表格外层容器 */
.table-wrapper {
  background-color: #fff;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
  width: 100%;
  max-width: 1000px;
  height: 320px;
  overflow-y: auto;
}

/* 表格标题（历史记录标题） */
.table-title {
  text-align: center;
  font-size: 20px;
  font-weight: 600;
  margin: 16px auto 20px;
  padding: 10px 24px;
  color: #333;
  background-color: #c7cbce;
  border-radius: 8px;
  border: 1px solid #a5a9ac;
  width: fit-content;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
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
  background-color: #e8ebf0;
  font-weight: 600;
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #ccc;
}

tbody td {
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
  word-break: break-word;
  color: #444;
}

/* 按钮样式 */
button {
  padding: 6px 14px;
  background-color: #409eff;
  color: #fff;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.25s ease;
  font-size: 14px;
}

button:hover {
  background-color: #66b1ff;
}
</style>
