<template>
  <div class="content" v-if="section">
    <h2>{{ section.title }}</h2>
      <p><strong>Time:</strong> {{ section.time }}</p>
      <p><strong>Summary:</strong> {{ section.summary }}</p>
      <p><strong>Key Actions:</strong></p>

    <ul>
      <li v-for="(action, idx) in section.actions" :key="idx">{{ action }}</li>
    </ul>

      <p><strong>Decisions:</strong></p>
    <ul>
      <li v-for="(item, idx) in section.decisions" :key="idx">{{ item }}</li>
    </ul>

      <p><strong>Conflicts of Interest:</strong></p>
    <ul>
      <li v-for="(conflict, idx) in section.conflicts" :key="idx">{{ conflict }}</li>
    </ul>

  </div>
  <div v-else>
    <p>Nothing</p>
  </div>
</template>

<script setup lang="ts">
import type { MeetingRecord } from '@/types/interface'
import {toRefs} from "vue"

defineProps<{
  section: MeetingRecord
}>()
</script>

<style scoped>
/* 内容区域容器样式 */
/* .content {
  padding: 2rem;
  flex-grow: 1;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  line-height: 1.6;
  font-size: 16px;
  color: #333;
} */


.content {
  padding: 1.5rem;
  flex-grow: 1;
  border: 1px solid #dcdcdc;        /* 添加浅灰色边框 */
  border-radius: 8px;               /* 边角圆润 */
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); /* 轻微阴影增加层次感 */
  background-color: #fff;   
  font-size: 16px;    
  color: #333;    /* 背景白色，与阴影形成对比 */
}

/* 标题样式 */
h2 {
  margin-top: 0;
  font-size: 1.8rem;
  color: #007acc;
  border-bottom: 2px solid #e6f0f6;
  padding-bottom: 0.5rem;
}

/* 段落标签 */
p {
  /* margin外边距，上右下左 */
  margin: 1rem 0 0.5rem;
  /* 400:正常 500:中等，700加粗 */
  font-weight: 700; 
  font-size: large;

}

/* 列表样式 */
ul {
  margin: 0.5rem 0 1.5rem 1.2rem;
  padding: 0;
}

li {
  margin-bottom: 0.4rem;
  line-height: 1.5;
}
</style>