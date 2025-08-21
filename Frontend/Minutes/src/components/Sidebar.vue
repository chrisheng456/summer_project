<template>
  <div class="sidebar">
    <!-- Top section -->
    <div @click="$emit('select', -1)" class="level-0">Meeting Details</div>

    <div @click="$emit('open-trans')" class="level-0">Transcription</div>

    <div class="level-1">Agenda</div>

    <!-- Section list -->
    <div
      v-for="(section, index) in sections"
      :key="index"
      :class="['level-2', { active: index === activeIndex }]"
      @click="$emit('select', index)"
    >
      <img src="@/assets/mic.png" class="item-icon" alt="mic" />
      <span class="item-title">{{ section.title }}</span>
    </div>

    <!-- Download dropdown -->
    <div class="download-area">
      <el-dropdown
        trigger="click"
        :teleported="true"
        popper-class="sidebar-dropdown"
        @command="onDownloadCommand"
      >
        <button class="download-btn">Download</button>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="pdf">Download PDF</el-dropdown-item>
            <el-dropdown-item command="docx">Download Word</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </div>
</template>

<script lang="ts" setup name="Sidebar">
import { defineProps, defineEmits } from 'vue'
import type { MeetingRecord } from '@/types/interface'

defineProps<{
  sections: MeetingRecord[]
  activeIndex: number
}>()

const emit = defineEmits<{
  (e: 'select', index: number): void
  (e: 'download', format: 'pdf' | 'docx'): void
  (e: 'open-trans'): void         
}>()

function onDownloadCommand(cmd: 'pdf' | 'docx') {
  emit('download', cmd)
}
</script>

<style scoped>
/* Sidebar container */
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
  font-size: 1.5rem;
  margin-bottom: 12px;
  padding-left: 0;
}


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

  transition: background-color 0.25s ease, box-shadow 0.25s ease, transform 0.12s ease;
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



.item-icon {
  font-size: 1rem;
  color: #007acc;
}


.item-title {
  color: #0f172a;
  font-weight: 600;
  font-size: 0.95rem;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.download-area {
  display: flex;
}


.download-btn {
  margin-left: 100px;
  padding: 6px 12px;
  background-color: #2078c6;
  border-radius: 5px;
  box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
  font-size: 1rem;
  margin-top: 50px;
  font-weight: 700;
  border: none;
  color: #fff;
  cursor: pointer;
}
.download-btn:hover {
  background-color: #e6f4ff;
  color: #333;
}

.level-0:hover {
  background-color: #e6f4ff;
  font-weight: 700;
  cursor: pointer;
}


:deep(.sidebar-dropdown) {
  z-index: 4000;
}

.item-icon {
  width: 18px;
  height: 18px;
  margin-right: 4px;
  vertical-align: middle;
}
</style>
