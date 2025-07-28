<template>
  <div class="layout">
    <!-- 侧边栏区域 -->
    <aside class="sidebar">
      <Sidebar
        :sections="sections"
        :activeIndex="activeIndex"
        @select="handleSelect"
      />
    </aside>

    <main class="content">
      <!-- 如果未选择区域，显示摘要 -->
      <MeetingAbstract v-if="activeIndex === -1" :abstract="defaultAbstract" />

      <!-- 选中区域，显示内容 -->
      <SectionContent
        v-else-if="sections.length"
        :section="sections[activeIndex]"
      />

      <!-- AI交互区域 -->
      <div class="question-header">
        <h3 style="display:inline-block;">Do you have any questions for AI?</h3>
        <button class="toggle-btn" @click="showInput = !showInput">
          {{ showInput ? '▼' : '▶' }}
        </button>
      </div>

      <div v-if="showInput" class="bottom-input-area">
        <!-- 用户输入指令 -->
        <textarea v-model="newContent" rows="4" placeholder="Enter your instruction (e.g., Please simplify the current summary)"></textarea>
        <button @click="submitContent" :disabled="loading">{{ loading ? 'Processing...' : 'Submit' }}</button>

        <!-- 显示AI结果，允许应用 -->
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
import SectionContent from '@/components/SectionContent.vue'
import Sidebar from '@/components/Sidebar.vue'
import MeetingAbstract from '@/components/MeetingAbstract.vue'
import { ref, onMounted } from 'vue'
import { fetchMeetingsData } from '@/api/fetchMeetingData'
import type { MeetingRecord } from '@/types/interface'

const sections = ref<MeetingRecord[]>([])
const activeIndex = ref(-1)
const defaultAbstract = ref('')
const newContent = ref('')
const showInput = ref(false)
const aiReply = ref('')
const loading = ref(false)

// 页面加载时，获取会议数据
onMounted(async () => {
  const result = await fetchMeetingsData()
  sections.value = result.meetings
  defaultAbstract.value = result.abstract || 'No abstract available.'
})

// 切换section
function handleSelect(index: number) {
  activeIndex.value = index
}

/**
 * 让AI对当前选中内容进行智能处理
 */
async function submitContent() {
  if (!newContent.value.trim()) {
    alert('请输入你的AI指令')
    return
  }
  loading.value = true

  // 1. 获取AI要处理的内容
  //    - 如果没选区块，就是摘要
  //    - 已选区块，可以拓展处理不同属性（此处以summary为例）
  let context = ''
  if (activeIndex.value === -1) {
    context = defaultAbstract.value
  } else if (sections.value[activeIndex.value]) {
    // 你可按需切换为 summary/keyActions/decisions 等
    context = sections.value[activeIndex.value].summary || ''
  }

  // 2. 组织prompt
const messages = [
  { role: 'system', content: '你是一个会议内容智能优化助手。' },
  { role: 'user', content: context },
  { role: 'user', content: newContent.value }
]

  try {
    // 3. 调用后端
    const response = await fetch('http://127.0.0.1:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages })
    })
    const data = await response.json()
    aiReply.value = data.reply
  } catch (error) {
    aiReply.value = ''
    alert('调用AI失败，请检查后端服务')
  } finally {
    loading.value = false
  }
}

/**
 * 应用AI结果：更新当前摘要或当前section内容
 */
function applyAIContent() {
  if (!aiReply.value) return

  if (activeIndex.value === -1) {
    defaultAbstract.value = aiReply.value
  } else if (sections.value[activeIndex.value]) {
    // 这里以 summary 为例，实际可按需支持更多字段
    sections.value[activeIndex.value].summary = aiReply.value
  }

  // 清空状态
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

h3 {
  margin: 0;
}

/* 切换按钮样式 */
.toggle-btn {
  margin-left: 10px;
  border: none;
  font-size: 18px;
  cursor: pointer;
  outline: none;
}

/* 底部输入区样式 */
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

button:hover {
  background-color: #66b1ff;
}

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