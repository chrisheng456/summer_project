<template>
  <div class="abstract">
    <h2>Meeting Overview</h2>

    <button class="edit-btn" @click="toggleEdit">
      {{ isEditing ? 'Save' : 'Edit' }}
    </button>

    <button class="reset-btn" @click="resetEdit">
      Reset
    </button>        

    <div v-if="isEditing">
      <textarea v-model="editableAbstract" class="input" />
    </div>


    <!-- Display mode: vertical list -->
    <div v-else class="abstract-list" :class="{ 'flash': justReset }">
      <div v-for="(row, i) in rows" :key="i" class="row">
        <template v-if="row.isPair">
          <span class="label">{{ row.key }}</span>
          <span class="value">{{ row.value }}</span>
        </template>
        <template v-else>
          <span class="value">{{ row.raw }}</span>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts" name="MeetingAbstract">
import { ref, watch, computed } from 'vue'

const props = defineProps<{ abstract: string }>()
const isEditing = ref(false)
const editableAbstract = ref('')
const justReset = ref(false)

watch(
  () => props.abstract,
  (val) => { editableAbstract.value = val },
  { immediate: true }
)

function toggleEdit() {
  if (isEditing.value) {
    console.log('保存 abstract:', editableAbstract.value)
  }
  isEditing.value = !isEditing.value
}

function resetEdit() {
  editableAbstract.value = props.abstract
  isEditing.value = false              
  justReset.value = true             
  setTimeout(() => (justReset.value = false), 600)
}


// Parse line by line, prioritize splitting at the first colon
const rows = computed(() => {
  const lines = (editableAbstract.value || '')
    .split(/\r?\n/)
    .map(s => s.trim())
    .filter(Boolean)

  return lines.map(raw => {
    const idx = raw.indexOf(':')
    if (idx > 0) {
      const key = raw.slice(0, idx).trim()
      const value = raw.slice(idx + 1).trim()
      if (key && value) return { isPair: true, key, value, raw }
    }
    return { isPair: false, raw }
  })
})
</script>

<style scoped>
.abstract {
  padding: 1.5rem 2rem;
  border: 1px solid #dcdcdc;
  background-color: #ffffff;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
  font-size: 16px;
  color: #333;
  line-height: 1.6;
  position: relative;
}

h2 {
  margin-top: 0;
  font-size: 1.8rem;
  color: #007acc;
  border-bottom: 2px solid #e6f0f6;
  padding-bottom: 0.5rem;
}


.input {
  width: 100%;
  padding: 8px;
  font-size: 16px;
  border: 1px solid #ccc;
  border-radius: 4px;
  resize: vertical;
  min-height: 100px;
}

.edit-btn {
  position: absolute;
  top: 1.5rem;
  right: 2rem;
  background-color: #007acc;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
}
.edit-btn:hover { background-color: #005fa3; }

.reset-btn {
  position: absolute;
  top: 1.5rem;
  right: 5rem;
  background-color: #007acc;
  color: white;
  border: none;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
}
.reset-btn:hover { background-color: #005fa3; }


.abstract-list {
  margin-top: 1rem;
  display: grid;
  gap: 10px;
}

.row {
  display: grid;
  grid-template-columns: 180px 1fr;
  gap: 12px;
  align-items: start;
  padding: 10px 12px;
  background: #f9fcff;
  border: 1px solid #e3e9f0;
  border-radius: 8px;
}


.row:not(:has(.label)) {
  grid-template-columns: 1fr;
}

.label {
  font-weight: 700;
  color: #0f172a;
}

.value {
  color: #334155;
  line-height: 1.7;
  word-break: break-word;
}


.flash {
  animation: flashBg .6s ease;
}
@keyframes flashBg {
  0%   { box-shadow: 0 0 0 0 rgba(17, 98, 255, 0.25); }
  100% { box-shadow: 0 0 0 0 rgba(17, 98, 255, 0); }
}


@media (max-width: 640px) {
  .row { grid-template-columns: 120px 1fr; }
}
</style>
