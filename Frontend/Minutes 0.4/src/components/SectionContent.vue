<template>
  <div class="content" v-if="editableSection">
    <h2>{{ editableSection.title }}</h2>

    <button class="edit-btn" @click="toggleEdit">
      {{ isEditing ? 'Save' : 'Edit' }}
    </button>

    <p><strong>Time:</strong></p>
    <div v-if="isEditing">
      <input v-model="editableSection.time" class="input" />
    </div>
    <div v-else>{{ editableSection.time }}</div>

    <p><strong>Summary:</strong></p>
    <div v-if="isEditing">
      <textarea v-model="editableSection.summary" class="input textarea" />
    </div>
    <div v-else>{{ editableSection.summary }}</div>

    <p><strong>Key Actions:</strong></p>
    <ul>
      <li v-for="(action, idx) in editableSection.actions" :key="idx">
        <div v-if="isEditing">
          <input v-model="editableSection.actions[idx]" class="input" />
        </div>
        <div v-else>{{ action }}</div>
      </li>
    </ul>

    <p><strong>Decisions:</strong></p>
    <ul>
      <li v-for="(item, idx) in editableSection.decisions" :key="idx">
        <div v-if="isEditing">
          <input v-model="editableSection.decisions[idx]" class="input" />
        </div>
        <div v-else>{{ item }}</div>
      </li>
    </ul>

    <p><strong>Conflicts of Interest:</strong></p>
    <ul>
      <li v-for="(conflict, idx) in editableSection.conflicts" :key="idx">
        <div v-if="isEditing">
          <input v-model="editableSection.conflicts[idx]" class="input" />
        </div>
        <div v-else>{{ conflict }}</div>
      </li>
    </ul>
  </div>
  <div v-else>
    <p>Nothing</p>
  </div>
</template>


<script setup lang="ts" name ="SectionContent">
import type { MeetingRecord } from '@/types/interface'
import { ref, watch } from 'vue'

const props = defineProps<{ section: MeetingRecord }>()

// 编辑模式切换
const isEditing = ref(false)
const editableSection = ref<MeetingRecord | null>(null)

watch(
  () => props.section,
  (newVal) => {
    editableSection.value = JSON.parse(JSON.stringify(newVal))
  },
  { immediate: true }
)


function toggleEdit() {
  if (isEditing.value) {
    console.log('保存数据：', editableSection.value)
    // TODO: emit 或请求 API
  }
  isEditing.value = !isEditing.value
}
</script>

<style scoped>
.content {
  padding: 1.5rem;
  flex-grow: 1;
  border: 1px solid #dcdcdc;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  background-color: #fff;
  font-size: 16px;
  color: #333;
}

h2 {
  margin-top: 0;
  font-size: 1.8rem;
  color: #007acc;
  border-bottom: 2px solid #e6f0f6;
  padding-bottom: 0.5rem;
}

p {
  margin: 1rem 0 0.5rem;
  font-weight: 700;
  font-size: large;
}

ul {
  margin: 0.5rem 0 1.5rem 1.2rem;
  padding: 0;
}

li {
  margin-bottom: 0.4rem;
  line-height: 1.5;
}

.input {
  width: 95%;
  padding: 6px;
  border: 1px solid #ccc;
  border-radius: 4px;
}

.textarea {
  min-height: 80px;
  resize: vertical;
}

.edit-btn {
  float: right;
  margin-top: -2.5rem;
  margin-right: 1rem;
  padding: 6px 12px;
  border: none;
  background-color: #007acc;
  color: white;
  border-radius: 4px;
  cursor: pointer;
}

.edit-btn:hover {
  background-color: #005fa3;
}
</style>