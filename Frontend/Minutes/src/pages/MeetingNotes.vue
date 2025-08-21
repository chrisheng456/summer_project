<template>
  <div class="layout">
    <!-- Sidebar -->
    <aside class="sidebar">
     <Sidebar
  :sections="sections"
  :activeIndex="activeIndex"
  @select="handleSelect"
  @download="handleDownload"
  @open-trans="goTransPage"   
/>

    </aside>

    <!-- Main content -->
    <main class="content">
      <!-- Back button -->
      <div class="back-btn">
        <button @click="goBack">Back</button>
      </div>

      <!-- Show abstract if no section selected -->
      <MeetingAbstract v-if="activeIndex === -1" :abstract="defaultAbstract" />

      <!-- Show details of selected section -->
      <SectionContent
        v-else-if="sections.length"
        :section="sections[activeIndex]"
      />

      <!-- AI interaction -->
      <div class="question-header">
        <h3 style="display:inline-block;">Do you have any questions for AI?</h3>
        <button class="toggle-btn" @click="showInput = !showInput">
          {{ showInput ? '▼' : '▶' }}
        </button>
      </div>

      <div v-if="showInput" class="bottom-input-area">
        <!-- User input -->
        <textarea
          v-model="newContent"
          rows="4"
          placeholder="Enter your instruction (e.g., Please simplify the current summary)"
        />
        <button @click="submitContent" :disabled="loading">
          {{ loading ? 'Processing...' : 'Submit' }}
        </button>

        <!-- AI reply with option to apply -->
        <div v-if="aiReply" class="ai-reply-box">
          <h4>AI Suggestion:</h4>
          <div class="ai-reply-text">{{ aiReply }}</div>
          <button @click="applyAIContent">Apply to Page</button>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts" name="MeetingNotes">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'  
import { ElMessage } from 'element-plus'

import Sidebar from '@/components/Sidebar.vue'
import SectionContent from '@/components/SectionContent.vue'
import MeetingAbstract from '@/components/MeetingAbstract.vue'

import { fetchMeetingsData } from '@/api/fetchMeetingsData'
import { meetingApi } from '@/api/modules'
import type { MeetingRecord } from '@/types/interface'

/** ===================== state ===================== */
const sections = ref<MeetingRecord[]>([])
const activeIndex = ref(-1)
const defaultAbstract = ref('')

const showInput = ref(false)
const newContent = ref('')
const aiReply = ref('')
const loading = ref(false)

const route = useRoute()
const router = useRouter()   

/** Convert query param to string */
function asString(v: unknown): string | undefined {
  if (Array.isArray(v)) return v[0] as string
  if (typeof v === 'string') return v
  return undefined
}

/** On page load */
onMounted(async () => {
  let schemeId = asString(route.query.scheme_id)
  let meetingId = asString(route.query.meeting_id)

  if (!schemeId || !meetingId) {
    const key = localStorage.getItem('selectedMeetingKey') || ''
    if (key.includes(':')) {
      const [sid, mid] = key.split(':')
      schemeId = sid
      meetingId = mid
    }
  }

  if (!schemeId || !meetingId) {
    ElMessage.error('Missing scheme_id / meeting_id, please open from Upload page.')
    sections.value = []
    defaultAbstract.value = 'No abstract available.'
    return
  }

  const result = await fetchMeetingsData(schemeId, meetingId)
  sections.value = result.meetings
  defaultAbstract.value = result.abstract || 'No abstract available.'
})

/** Handle section selection */
function handleSelect(index: number) {
  activeIndex.value = index
}

/** Back button logic */
function goBack() {
  router.push({ name: 'UploadHistory' }) 
}



/** Get schemeId / meetingId */
function getIds() {
  let schemeId = asString(route.query.scheme_id)
  let meetingId = asString(route.query.meeting_id)

  if (!schemeId || !meetingId) {
    const key = localStorage.getItem('selectedMeetingKey') || ''
    if (key.includes(':')) {
      const [sid, mid] = key.split(':')
      schemeId = schemeId || sid
      meetingId = meetingId || mid
    }
  }
  return { schemeId, meetingId }
}

function goTransPage() {
  const { schemeId, meetingId } = getIds() 
  if (!schemeId || !meetingId) return
  router.push({
    name: 'TransPage',
    query: { scheme_id: schemeId, meeting_id: meetingId }
  })
}

/** Download blob file */
function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

/** Handle download request from Sidebar */
async function handleDownload(format: 'pdf' | 'docx') {
  const { schemeId, meetingId } = getIds()
  if (!schemeId || !meetingId) {
    ElMessage.error('Missing scheme_id / meeting_id')
    return
  }

  try {
    let blob: Blob
    if (format === 'pdf') {
      blob = await meetingApi.exportPdf(schemeId, meetingId)
    } else {
      blob = await meetingApi.exportDocx(schemeId, meetingId)
    }

    const filename = `meeting-${meetingId}.${format}`
    downloadBlob(blob, filename)
    ElMessage.success(`Start downloading ${filename}`)
  } catch (err) {
    console.error('[handleDownload] failed', err)
    ElMessage.error('Download failed, please try again.')
  }
}


async function submitContent() {
  const prompt = newContent.value.trim()
  if (!prompt) {
    ElMessage.warning('Please enter your AI instruction')
    return
  }

  let context = ''
  if (activeIndex.value === -1) {
    context = defaultAbstract.value
  } else if (sections.value[activeIndex.value]) {
    context = sections.value[activeIndex.value].summary || ''
  }

  const messages = [
  {
    role: 'system',
    content:
      'You are an assistant that improves meeting notes. Always respond in English unless explicitly asked otherwise.'
  },
  { role: 'user', content: `Context:\n${context}` },
  { role: 'user', content: newContent.value }
]

 try {
    loading.value = true
    const resp = await fetch('/ai-api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages })
    })
    const data = await resp.json()
    aiReply.value = data.reply
  } catch (err) {
    aiReply.value = ''
    ElMessage.error('AI failed, please check the backend service.')
  } finally {
    loading.value = false
  }
}
function applyAIContent() {
  if (!aiReply.value) return
  if (activeIndex.value === -1) {
    defaultAbstract.value = aiReply.value
  } else if (sections.value[activeIndex.value]) {
    sections.value[activeIndex.value].summary = aiReply.value
  }
  aiReply.value = ''
  newContent.value = ''
}
</script>

<style scoped>

.layout {
  display: flex;
  height: 100vh;
}


.sidebar {
  width: 240px;
  background-color: #f4f4f4;
  border-right: 1px solid #ccc;
  overflow-y: auto;
}


.content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}


.back-btn {
  margin-bottom: 15px;
}
.back-btn button {
  background-color: #4884d3;
  border: none;
  color: white;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
}
.back-btn button:hover {
  background-color: #444;
}

.question-header {
  display: flex;
  align-items: center;
  margin-top: 20px;
  margin-bottom: 10px;
}
h3 { margin: 0; }


.toggle-btn {
  margin-left: 10px;
  border: none;
  font-size: 18px;
  cursor: pointer;
  outline: none;
}


.bottom-input-area {
  margin-top: 10px;
  border-top: 1px solid #ddd;
  padding-top: 10px;
}
textarea {
  width: 99%;
  box-sizing: border-box;
  margin-bottom: 10px;
  padding: 8px;
  border-radius: 6px;
  border: 1px solid #ccc;
}

button {
  padding: 6px 12px;
  cursor: pointer;
  background-color: #2b76c2;
  border: none;
  border-radius: 6px;
  color: white;
}
button:hover { background-color: #66b1ff; }


.ai-reply-box {
  margin-top: 15px;
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background-color: #f9f9f9;
}
.ai-reply-text {
  white-space: pre-wrap;
  line-height: 1.6;
  margin-bottom: 10px;
}
</style>
