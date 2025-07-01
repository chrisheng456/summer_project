<template>
  <div class="sidebar">
    <div
      v-for="(section, index) in sections"
      :key="index"
      :class="{ active: index === activeIndex }"
      @click="$emit('select', index)"
    >
      {{ section.title }}
    </div>
  </div>
</template>

<script lang="ts" setup>
import { defineProps } from 'vue'
import type { MeetingRecord} from '@/types/interface'

defineProps<{
sections: MeetingRecord[]
activeIndex: number
}>()
</script>

<style scoped>
.sidebar {
  width: 220px;
  background-color: #f0f0f0;
  border-right: 1px solid #ccc;
  padding: 1rem;
}
.sidebar div {
  margin: 10px 0;
  cursor: pointer;
  padding: 6px;
}
.sidebar div:hover,
.sidebar .active {
  background-color: #dceeff;
  font-weight: bold;
}
</style>
