<template>
  <div>
    <HeaderBar />
    <div class="container">
      <!-- 上传音频 -->
      <div class="upload-section">
        <h3>Upload New Audio</h3>

        <el-upload
          class="upload-card"
          drag
          :auto-upload="false"
          v-model:file-list="fileList"
          :on-change="onFileChange"
          accept=".mp3,.wav,.m4a"
          :limit="1"
        >
          <el-icon class="upload-icon"><UploadFilled /></el-icon>
          <div class="el-upload__text">
            Drop audio file here or <em>click to upload</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">Only mp3/wav/m4a files</div>
          </template>
        </el-upload>

        <!-- 处理按钮（选了会议 + 选了音频 才能点击） -->
        <div class="process-bar">
          <button
            class="process-btn"
            :disabled="!selectedKey || !selectedFile || processing"
            @click="startAnalyze"
          >
            {{ processing ? 'Processing…' : 'Process & Analyze' }}
          </button>
          <span class="hint" v-if="!selectedKey || !selectedFile">
            Select a meeting and choose an audio file to enable processing.
          </span>
        </div>
      </div>

      <!-- 会议列表 -->
      <div class="history-section">
        <h3 class="table-title">Meetings History</h3>
        <div class="table-wrapper">
          <table>
            <thead>
              <tr>
                <th style="width:60px;">Select</th>
                <th>Title</th>
                <th>Date</th>
                <th>Scheme</th>
                <th>Status</th>
                <th>Details</th>
                <th class="action-column"></th>
              </tr>
            </thead>

            <tbody>
              <tr
                v-for="item in history"
                :key="rowKey(item)"
              >
                <!-- 单选：选择一个会议 -->
                <td>
                  <input
                    type="radio"
                    name="meetingRadio"
                    :value="rowKey(item)"
                    v-model="selectedKey"
                  />
                </td>

                <td>{{ item.title }}</td>
                <td>{{ formatDate(item.date) }}</td>
                <td>{{ item.scheme_name }}</td>

                <!-- 简单状态：Done/Processing/None -->
                <td>
                  <span v-if="processingKey === rowKey(item)">Processing…</span>
                  <span v-else-if="isAnalyzed(item)">Done</span>
                  <span v-else>None</span>
                </td>

                <td>
                  <button
                    :disabled="!isAnalyzed(item)"
                    @click="viewDetail(item)"
                  >
                    View Details
                  </button>
                </td>
              </tr>

              <tr v-if="!history.length">
                <td colspan="7" style="text-align:center; color:#888; padding:20px;">
                  No meetings
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
import { ref, onMounted, watch, computed } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import HeaderBar from '@/components/HeaderBar.vue'
import { meetingApi } from '@/api/modules'
import type { UploadFile } from 'element-plus'
import type { Meetings, MeetingDetail } from '@/types'
import { useMeetingStore } from '@/stores/meeting'
import { adaptApiResultToMeetingDetail } from '@/api/transform/adaptMeeting'

// Pinia
const meetingStore = useMeetingStore()

// 会议列表
const history = ref<Meetings[]>([])
const router = useRouter()

// 1.读取本地会议 + 恢复“上次选中的会议”
onMounted(() => {
  try {
    const raw = localStorage.getItem('meetings')
    history.value = raw ? JSON.parse(raw) : []
  } catch {
    history.value = []
  }

  const last = localStorage.getItem('selectedMeetingKey')
  if (last) selectedKey.value = last
})

// 行 key
const rowKey = (m: Meetings) => `${m.scheme_id}:${m.meeting_id}`

// 当前选中的会议 key
const selectedKey = ref<string>('')

// 同步本地存储
watch(selectedKey, v => {
  localStorage.setItem('selectedMeetingKey', v ?? '')
})

// 当前选择的会议对象
const selectedMeeting = computed<Meetings | null>(() => {
  if (!selectedKey.value) return null
  const [sid, mid] = selectedKey.value.split(':')
  return history.value.find(
    m => String(m.scheme_id) === sid && String(m.meeting_id) === mid
  ) || null
})

// 选择的音频（仅通过 el-upload）
const selectedFile = ref<File | null>(null)
const fileList = ref<UploadFile[]>([])
function onFileChange(file: UploadFile) {
  selectedFile.value = (file?.raw as File) || null
}

// 状态：哪个会议正在处理
const processingKey = ref<string>('')
const processing = ref(false)

// ✅ 是否“已处理完成”用 Pinia
function isAnalyzed(item: Meetings) {
  return meetingStore.isAnalyzed(item.scheme_id, item.meeting_id)
}

// 开始处理：送到 /pipeline/analyze，返回结果存 Pinia
async function startAnalyze() {
  const m = selectedMeeting.value
  if (!m) {
    ElMessage.warning('Please select a meeting first')
    return
  }
  if (!selectedFile.value) {
    ElMessage.warning('Please choose an audio file')
    return
  }

  try {
    processing.value = true
    processingKey.value = rowKey(m)

    const detail = await meetingApi.analyze(
      selectedFile.value,
      String(m.scheme_id),
      String(m.meeting_id)
    ) as MeetingDetail

    const adapted = adaptApiResultToMeetingDetail(detail)
    meetingStore.setDetail(adapted, m.scheme_id, m.meeting_id)

    console.group('[UploadHistory.startAnalyze]')
    console.log('selectedKey =', selectedKey.value)
    console.log('rowKey =', rowKey(m))
    console.log('isAnalyzed?', meetingStore.isAnalyzed(m.scheme_id, m.meeting_id))
    console.log('store.getByIds =', meetingStore.getByIds(m.scheme_id, m.meeting_id))
    console.groupEnd()

    ElMessage.success('Analysis completed')
  } catch (err: any) {
    ElMessage.error(err?.response?.data?.message || 'Analyze failed, please try again')
  } finally {
    processing.value = false
    processingKey.value = ''
    // 清空已选文件（如需保留可去掉）
    fileList.value = []
    selectedFile.value = null
  }
}

// 查看详情（只在已处理完成后可点）
function viewDetail(item: Meetings) {
  if (!isAnalyzed(item)) return
  router.push({
    path: '/MeetingNotes',
    query: {
      scheme_id: String(item.scheme_id),
      meeting_id: String(item.meeting_id),
    },
  })
}

// 时间格式
function formatDate(iso: string) {
  try { return new Date(iso).toLocaleDateString() } catch { return iso }
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
  font-family: 'Helvetica Neue', Arial, sans-serif;
}

/* Upload card */
.upload-section {
  background-color: #fff;
  padding: 20px 20px;
  margin-top: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.337);
  width: 100%;
  max-width: 1000px;
  margin-bottom: 24px;
  text-align: center;
}

.upload-section:hover {
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.2);
  transform: translateY(-4px);
  transition: all 0.3s ease;
}

.upload-section h3 {
  margin-bottom: 16px;
  font-size: 1.5rem;
  color: #333;
  font-weight: 600;
}

/* Process bar & button */
.process-bar {
  margin-top: 16px;
  display: flex;
  gap: 12px;
  justify-content: center;
  align-items: center;
}

.process-btn {
  padding: 10px 18px;
  background-color: #3b82f6;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(59, 130, 246, 0.35);
}

.process-btn[disabled] {
  opacity: 0.6;
  cursor: not-allowed;
}

.hint {
  color: #888;
  font-size: 13px;
}

/* Table */
.table-wrapper {
  background-color: #fff;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.337);
  width: 100%;
  max-width: 1000px;
  height: 420px;
  overflow-y: auto;
}

.table-title {
  text-align: center;
  font-size: 1.5rem;
  font-weight: 600;
  margin: 16px auto 10px;
  padding: 10px 24px;
  color: #333;
  width: fit-content;
}

table {
  width: 100%;
  border-collapse: collapse;
  table-layout: fixed;
  font-size: 15px;
}

thead th {
  position: sticky;
  top: 0;
  background-color: #fff;
  font-weight: 700;
  padding: 14px 12px;
  text-align: left;
  border-bottom: 1.5px solid #eee;
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
  background-color: #f0f8ff;
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

/* Buttons */
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

button[disabled] {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Responsive */
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

.action-column {
  width: 10px;
  max-width: 15px;
  text-align: left;
  padding: 0 40px 0 0;
}

.dots-button {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: transparent;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: background-color 0.2s ease;
  font-size: 18px;
  color: #666;
}

.dots-button:hover {
  background-color: #e0e6ed;
}
</style>
