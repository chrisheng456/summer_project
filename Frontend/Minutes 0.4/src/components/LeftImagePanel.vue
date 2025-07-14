<template>
  <div class="left-panel" :style="backgroundStyle">
    <div class="image-wrapper">
      <slot></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// 接收 props
const props = defineProps<{ imageUrl: string }>()

// 动态背景样式，防止图片路径被错处理
const backgroundStyle = computed(() => ({
  backgroundImage: `url(${props.imageUrl})`,
  backgroundSize: 'cover',
  backgroundPosition: 'center',
}))
</script>

<style scoped>
.left-panel {
  width: 60vw;
  height: 100vh;
}

/* 遮罩 + 内容容器 */
.image-wrapper {
  width: 100%;
  height: 100%;
  background-color: transparent; 
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
}
</style>
