<template>
  <div class="layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <Sidebar
        :sections="sections"
        :activeIndex="activeIndex"
        @select="handleSelect"
      />
    </aside>

    <!-- 主内容 -->
    <main class="content">
      <!-- 未选择任何 section 时显示摘要 -->
      <MeetingAbstract v-if="activeIndex === -1" :abstract="defaultAbstract" />

      <!-- 选中某个 section 时显示详情 -->
      <SectionContent
        v-else-if="sections.length"
        :section="sections[activeIndex]"
      />

      <!-- AI 交互 -->
      <div class="question-header">
        <h3 style="display:inline-block;">Do you have any questions for AI?</h3>
        <button class="toggle-btn" @click="showInput = !showInput">
          {{ showInput ? '▼' : '▶' }}
        </button>
      </div>

      <div v-if="showInput" class="bottom-input-area">
        <!-- 用户输入指令 -->
        <textarea
          v-model="newContent"
          rows="4"
          placeholder="Enter your instruction (e.g., Please simplify the current summary)"
        />
        <button @click="submitContent" :disabled="loading">
          {{ loading ? 'Processing...' : 'Submit' }}
        </button>

        <!-- 显示 AI 结果，允许应用 -->
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
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'

import Sidebar from '@/components/Sidebar.vue'
import SectionContent from '@/components/SectionContent.vue'
import MeetingAbstract from '@/components/MeetingAbstract.vue'

import { fetchMeetingsData } from '@/api/fetchMeetingsData'
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

/** 把 query 值规范为 string（可能是 string | string[] | undefined） */
function asString(v: unknown): string | undefined {
  if (Array.isArray(v)) return v[0] as string
  if (typeof v === 'string') return v
  return undefined
}

/** 页面加载：从路由或本地兜底拿到 schemeId/meetingId，并拉取数据 */
onMounted(async () => {
  // 1) 优先从路由 query 取
  let schemeId = asString(route.query.scheme_id)
  let meetingId = asString(route.query.meeting_id)

  // 2) 刷新导致 query 丢失时，从本地兜底（Upload 页保存过）
  if (!schemeId || !meetingId) {
    const key = localStorage.getItem('selectedMeetingKey') || ''
    if (key.includes(':')) {
      const [sid, mid] = key.split(':')
      schemeId = sid
      meetingId = mid
    }
  }

  // 3) 仍然拿不到则提示
  if (!schemeId || !meetingId) {
    ElMessage.error('Missing scheme_id / meeting_id, please open from Upload page.')
    sections.value = []
    defaultAbstract.value = 'No abstract available.'
    return
  }

  // 4) 拉取并渲染（从 Pinia 读原始数据并做修剪）
  const result = await fetchMeetingsData(schemeId, meetingId)
  sections.value = result.meetings
  defaultAbstract.value = result.abstract || 'No abstract available.'
})

/** 选择侧边栏的 section */
function handleSelect(index: number) {
  activeIndex.value = index
}

/** 让 AI 改写摘要或当前 section 的 summary */
async function submitContent() {
  const prompt = newContent.value.trim()
  if (!prompt) {
    ElMessage.warning('请输入你的 AI 指令')
    return
  }

  // 取上下文：未选中则用摘要，已选中则用当前 section 的 summary
  let context = ''
  if (activeIndex.value === -1) {
    context = defaultAbstract.value
  } else if (sections.value[activeIndex.value]) {
    context = sections.value[activeIndex.value].summary || ''
  }

  const messages = [
    { role: 'system', content: '你是一个会议内容智能优化助手。' },
    { role: 'user', content: context },
    { role: 'user', content: prompt }
  ]

  try {
    loading.value = true
    const resp = await fetch('http://127.0.0.1:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages })
    })
    const data = await resp.json()
    aiReply.value = data.reply
  } catch (err) {
    aiReply.value = ''
    ElMessage.error('调用 AI 失败，请检查后端服务')
  } finally {
    loading.value = false
  }
}

/** 应用 AI 结果到页面（摘要或当前 section 的 summary） */
function applyAIContent() {
  if (!aiReply.value) return

  if (activeIndex.value === -1) {
    defaultAbstract.value = aiReply.value
  } else if (sections.value[activeIndex.value]) {
    sections.value[activeIndex.value].summary = aiReply.value
  }

  // 清空本次交互
  aiReply.value = ''
  newContent.value = ''
}
</script>

<style scoped>
/* 页面基础布局 */
.layout {
  display: flex;
  height: 100vh;
}

/* 侧边栏样式 */
.sidebar {
  width: 240px;
  background-color: #f4f4f4;
  border-right: 1px solid #ccc;
  overflow-y: auto;
}

/* 主内容区域 */
.content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

/* 问题区域标题 */
.question-header {
  display: flex;
  align-items: center;
  margin-top: 20px;
  margin-bottom: 10px;
}
h3 { margin: 0; }

/* 切换按钮 */
.toggle-btn {
  margin-left: 10px;
  border: none;
  font-size: 18px;
  cursor: pointer;
  outline: none;
}

/* 底部输入区 */
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
  background-color: #409eff;
  border: none;
  border-radius: 6px;
  color: white;
}
button:hover { background-color: #66b1ff; }

/* AI 结果展示 */
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