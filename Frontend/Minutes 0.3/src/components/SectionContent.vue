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
    <p>暂无内容</p>
  </div>
</template>

<script setup lang="ts">
import type { MeetingRecord } from '@/types/interface'

defineProps<{
  section: MeetingRecord
}>()
</script>

<style scoped>
.content {
  padding: 1.5rem;
  flex-grow: 1;
}
h2 {
  margin-top: 0;
}
</style>