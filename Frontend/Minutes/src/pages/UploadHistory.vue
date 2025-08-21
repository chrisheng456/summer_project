<template>
  <div>
    <HeaderBar />
    <div class="container">
      <!-- Upload audio -->
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

        <!-- Process button (enabled only when a meeting + audio file are selected) -->
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


      <!-- Meeting list -->
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

// Meeting list
const history = ref<Meetings[]>([])
const router = useRouter()

// Load local meetings + restore "last selected meeting"
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


const rowKey = (m: Meetings) => `${m.scheme_id}:${m.meeting_id}`

// Currently selected meeting key
const selectedKey = ref<string>('')

// Sync with local storage
watch(selectedKey, v => {
  localStorage.setItem('selectedMeetingKey', v ?? '')
})


const selectedMeeting = computed<Meetings | null>(() => {
  if (!selectedKey.value) return null
  const [sid, mid] = selectedKey.value.split(':')
  return history.value.find(
    m => String(m.scheme_id) === sid && String(m.meeting_id) === mid
  ) || null
})


// Selected audio file (via el-upload)
const selectedFile = ref<File | null>(null)
const fileList = ref<UploadFile[]>([])
function onFileChange(file: UploadFile) {
  selectedFile.value = (file?.raw as File) || null
}

// State: which meeting is being processed
const processingKey = ref<string>('')
const processing = ref(false)

// Whether a meeting has been analyzed (via Pinia)
function isAnalyzed(item: Meetings) {
  return meetingStore.isAnalyzed(item.scheme_id, item.meeting_id)
}

// Start analyzing: send to /pipeline/analyze, save result to Pinia
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
    fileList.value = []
    selectedFile.value = null
  }
}


// View details (enabled only if analysis is done)
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

// Format date
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
