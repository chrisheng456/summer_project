
<template>
<div class="sidebar">
  <!-- 顶部内容 -->
  <div @click="$emit('select', -1)" class="level-0">Meeting Details</div>
  <div class="level-1">Agenda</div>

  <!-- Section 列表 -->
  <div
    v-for="(section, index) in sections"
    :key="index"
    :class="['level-2', { active: index === activeIndex }]"
    @click="$emit('select', index)"
  >
<!-- 当 index === activeIndex 成立时，为这个元素动态添加类名 active，vue中动态绑定class的语法-->
<!-- @click="$emit('select', index)"子传父的方法之一：自定义时间，在父组件中编辑事件“select” -->
<!-- 子组件点击事件，触发父组件的事件select，调用handleSelect函数，传入参数赋值给activeIndex -->
    {{ section.title }}
  </div>
</div>
</template>




<script lang="ts" setup name ="Sidebar">
import { defineProps } from 'vue'
import type { MeetingRecord} from '@/types/interface'

defineProps<{
sections: MeetingRecord[]
activeIndex: number
}>()
</script>

<style scoped>
/* 侧边栏整体样式 */
.sidebar {
  width: 220px;
  background-color: #f0f8ff;
  border-right: 1px solid #ddd;
  padding: 1rem;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
}



.level-0,
.level-1 {
  font-weight: 700;
  font-size: 1rem;
  margin-bottom: 12px;
  padding-left: 0;
}


/* 二级项：Section 列表 */
.level-2 {
  margin-bottom: 8px;
  padding: 6px 12px;
  padding-left: 20px; /* 缩进显示层级感 */
  border-radius: 6px;
  transition: background-color 0.3s ease, font-weight 0.3s ease;
  cursor: pointer;
}

/* 一级导航项：Meeting Details 和 Agenda */
.level-0:hover {
  background-color: #e6f4ff;
  font-weight: 700;
  cursor: pointer;
}

.level-2:hover {
  background-color: #e6f4ff;
  font-weight: 700;
}

.level-2.active {
  background-color: #cceeff;
  font-weight: 600;
  color: #007acc;
}

</style>
