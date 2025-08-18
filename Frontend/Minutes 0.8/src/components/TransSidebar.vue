<template>
  <div class="sidebar">
    <!-- Back 在侧边栏顶部 -->
    <button class="back-btn" @click="$emit('back')">Back</button>

    <!-- 顶部标题（点它回到“全部议题”） -->
    <div @click="$emit('select', -1)" class="level-0 clickable">Transcription</div>
    <div class="level-1">Agenda</div>

    <!-- 议题列表 -->
    <div
      v-for="(section, index) in sections"
      :key="index"
      :class="['level-2', { active: index === activeIndex }]"
      @click="$emit('select', index)"
    >
      <img src="@/assets/mic.png" class="item-icon" alt="mic" />
      <span class="item-title">{{ section.title }}</span>
    </div>
  </div>
</template>

<script lang="ts" setup name="TransSidebar">
import { defineProps, defineEmits } from 'vue'

type SideItem = { title: string }

defineProps<{
  sections: SideItem[]
  activeIndex: number
}>()

defineEmits<{
  (e: 'select', index: number): void
  (e: 'back'): void
}>()
</script>

<style scoped>
/* 侧边栏整体样式（去掉淡蓝底） */
.sidebar {
  width: 240px;
  background-color: #fff;            /* 原 #f0f8ff -> #fff */
  border-right: 1px solid #ddd;
  padding: 12px 14px;
  box-sizing: border-box;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
}

/* Back 按钮 */
.back-btn {
  width: 100%;
  background-color: #4884d3;
  color: #fff;
  border: none;
  border-radius: 8px;
  padding: 8px 12px;
  font-weight: 600;
  cursor: pointer;
  margin-bottom: 12px;
}
.back-btn:hover { background-color: #3b6eb0; }

/* 标题层级 */
.level-0,
.level-1 {
  font-weight: 700;
  font-size: 1.5rem;
  margin-bottom: 12px;
  padding-left: 0;
  color: #0f172a;
}
.clickable { cursor: pointer; }
.clickable:hover {
  background-color: transparent;      /* 不要浅蓝高亮 */
}

/* 列表卡片 */
.level-2 {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  padding: 10px 14px;
  background: #f8fafa;
  border: 1px solid #e0e6ed;
  border-radius: 10px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
  transition: background-color .25s, box-shadow .25s, transform .12s;
  cursor: pointer;
}
.level-2:hover {
  background: #f5f9ff;
  box-shadow: 0 4px 12px rgba(0, 80, 200, 0.15);
  transform: translateY(-2px);
}
.level-2.active {
  background: #e6f2ff;
  border-color: #4a90e2;
  box-shadow: 0 6px 14px rgba(0, 80, 200, 0.2);
}

/* 图标与标题 */
.item-icon {
  width: 18px;
  height: 18px;
  margin-right: 4px;
  vertical-align: middle;
}
.item-title {
  color: #0f172a;
  font-weight: 600;
  font-size: 0.95rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
