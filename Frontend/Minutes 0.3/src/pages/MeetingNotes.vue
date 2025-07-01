<template>
  <div class="layout">
    <aside class="sidebar">       <!-- 左侧导航区 -->
      <Sidebar
        :sections="sections"
        :activeIndex="activeIndex"
        @select="handleSelect"
      />
    </aside>

    <main class="content">        <!-- 右侧内容区 -->
      <SectionContent
        v-if="sections.length"
        :section="sections[activeIndex]"
      />
    </main>
  </div>
</template>

<script setup lang="ts" name ="MeetingNotes">
import SectionContent from '@/components/SectionContent.vue';
import Sidebar from '@/components/Sidebar.vue';
import {ref, onMounted} from 'vue'
import { fetchMeetingsData } from '@/api/fetchMeetingData'
import type { MeetingRecord } from '@/types/interface';

const sections = ref<MeetingRecord[]>([])//这里要显式声明下sections的类型是MeetingRecord数组
const activeIndex = ref(0)

onMounted(async () => {
 sections.value = await fetchMeetingsData()
})

function handleSelect(index: number) {
  activeIndex.value = index
}

</script>

<style scoped>
.layout {
  display: flex;               /* 启用 Flex 布局 */
  height: 100vh;               /* 让整体高度填满整个视口 */
}

.sidebar {
  width: 240px;                /* 左侧导航区宽度固定 */
  background-color: #f4f4f4;
  border-right: 1px solid #ccc;
  overflow-y: auto;
}

.content {
  flex: 1;                     /* 内容区占剩余空间 */
  padding: 20px;
  overflow-y: auto;
}

</style>