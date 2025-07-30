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
    <p v-else>{{ editableAbstract }}</p>
  </div>
</template>



<script setup lang="ts" name="MeetingAbstract">
import { ref, watch } from 'vue'

// 声明传入的 abstract 字符串
const props = defineProps<{ abstract: string }>()
const isEditing = ref(false)
const editableAbstract = ref('')

// 当 props.abstract 变化时，同步更新可编辑版本
watch(
  // ① 要侦听的内容
  () => props.abstract,

  // ② 回调函数：val 是最新值
  (val) => {
    editableAbstract.value = val
  },

  // ③ 选项对象
  { immediate: true }
)

function toggleEdit() {
  if (isEditing.value) {
    console.log('保存 abstract：', editableAbstract.value)
    // 你可以 emit 到父组件，也可以调用 API 保存
  }
  isEditing.value = !isEditing.value
}

function resetEdit() {
  editableAbstract.value = props.abstract
}


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

.edit-btn:hover {
  background-color: #005fa3;
}

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

.reset-btn:hover {
  background-color: #005fa3;
}
</style>