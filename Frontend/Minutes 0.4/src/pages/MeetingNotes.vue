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

      <!-- 如果选中了区域，显示区域内容 -->
      <SectionContent
        v-else-if="sections.length"
        :section="sections[activeIndex]"
      />

      <!-- 标题和切换按钮区域 -->
      <div class="question-header">
        <h3 style="display:inline-block;">你有什么想问AI的吗？</h3>
        <button class="toggle-btn" @click="showInput = !showInput">
          {{ showInput ? '▼' : '▶' }}
        </button>
      </div>

      <!-- 根据 showInput 控制显示或隐藏输入区 -->
      <div v-if="showInput" class="bottom-input-area">
        <textarea v-model="newContent" rows="4" placeholder="请输入内容（例如：请帮我简化一下summary）"></textarea>
        <button @click="submitContent">提交</button>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts" name ="MeetingNotes">
import SectionContent from '@/components/SectionContent.vue';     // 区域内容组件
import Sidebar from '@/components/Sidebar.vue';                   // 侧边栏组件
import MeetingAbstract from '@/components/MeetingAbstract.vue';  // 会议摘要组件
import {ref, onMounted} from 'vue';                              // Vue核心 API
import { fetchMeetingsData } from '@/api/fetchMeetingData';       // 获取会议数据接口
import type { MeetingRecord } from '@/types/interface';           // 数据类型

// 会议区域数据
const sections = ref<MeetingRecord[]>([]);
// 当前选中的区域索引，-1代表未选择
const activeIndex = ref(-1);
// 摘要内容
const defaultAbstract = ref('');
// AI提示内容
const newContent = ref('');
// 显示或隐藏输入框控制
const showInput = ref(false);

// 缓加数据
onMounted(async () => {
  const result = await fetchMeetingsData();
  sections.value = result.meetings;
  defaultAbstract.value = result.abstract || 'No abstract available.';
});

// 选中区域处理
function handleSelect(index: number) {
  activeIndex.value = index;
}

// 提交按钮处理（后端提交可扩展）
function submitContent() {
  alert(`提交内容：${newContent.value}`);
  newContent.value = '';
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
  background: none;
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
}

button {
  padding: 6px 12px;
  cursor: pointer;
}
</style>