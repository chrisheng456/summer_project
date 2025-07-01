<!-- src/components/MeetingAbstract.vue -->
<template>
  <div class="abstract">
    <h2>Meeting Overview</h2>
    <p>{{ abstract }}</p>

  </div>
</template>

<script setup lang="ts" name="MeetingAbstract">
import { defineProps } from 'vue'
defineProps<{abstract:string}>()
</script>

<style scoped>
.abstract {
  padding: 1.5rem 2rem;                /* 上下 1.5rem，左右 2rem 更舒适 */
  border: 1px solid #dcdcdc;           /* 更柔和的浅灰边框 */
  background-color: #ffffff;           /* 改为纯白背景，与主内容一致 */
  border-radius: 8px;                  /* 圆角 */
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);  /* 添加轻微阴影，增加层次感 */
  font-size: 16px;
  color: #333;                         /* 更适合正文的颜色 */
  line-height: 1.6;                    /* 行距更佳阅读 */
}
h2{
 margin-top: 0;
  font-size: 1.8rem;
  color: #007acc;
  border-bottom: 2px solid #e6f0f6;
  padding-bottom: 0.5rem;
}


</style>