<template>
  <div class="layout">
    <aside class="sidebar">
      <Sidebar
        :sections="sections"
        :activeIndex="activeIndex"
        @select="handleSelect"
      />
    </aside>

    <main class="content">
      <MeetingAbstract v-if="activeIndex === -1" :abstract="defaultAbstract" />
      <SectionContent
        v-else-if="sections.length"
        :section="sections[activeIndex]"
      />

      <!-- 标题部分 + 切换箭头按钮 -->
      <div class="question-header">
        <h3 style="display:inline-block;">你有什么想问AI的吗？</h3>
        <button class="toggle-btn" @click="showInput = !showInput">
          {{ showInput ? '▼' : '▶' }}
        </button>
      </div>

      <!-- 根据 showInput 控制显示隐藏 -->
      <div v-if="showInput" class="bottom-input-area">
        <textarea v-model="newContent" rows="4" placeholder="请输入内容（例如：请帮我简化一下summary）"></textarea>
        <button @click="submitContent">提交</button>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts" name ="MeetingNotes">
import SectionContent from '@/components/SectionContent.vue';
import Sidebar from '@/components/Sidebar.vue';
import MeetingAbstract from '@/components/MeetingAbstract.vue'
import {ref, onMounted} from 'vue'
import { fetchMeetingsData } from '@/api/fetchMeetingData'
import type { MeetingRecord } from '@/types/interface';

const sections = ref<MeetingRecord[]>([])
const activeIndex = ref(-1)
const defaultAbstract = ref('')
const newContent = ref('') // 输入内容
const showInput = ref(false) // 控制底部输入框显示/隐藏

onMounted(async () => {
  const result = await fetchMeetingsData()
  sections.value = result.meetings
  defaultAbstract.value = result.abstract || 'No abstract available.'
})

function handleSelect(index: number) {
  activeIndex.value = index
}

function submitContent() {
  alert(`提交内容：${newContent.value}`)
  // 你可以在这里加入后端提交
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

/* 标题和箭头按钮布局 */
.question-header {
  display: flex;
  align-items: center;
  margin-top: 20px;
  margin-bottom: 10px;
}

h3 {
  margin: 0;
}

.toggle-btn {
  margin-left: 10px;
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  outline: none;
}

/* 新增样式，用于底部输入区域 */
.bottom-input-area {
  margin-top: 10px;
  border-top: 1px solid #ddd;
  padding-top: 10px;
}
textarea {
  width: 99%;
  box-sizing: border-box;
  margin-bottom: 10px;
}
button {
  padding: 6px 12px;
  cursor: pointer;
}
</style>
