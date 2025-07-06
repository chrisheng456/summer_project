<template>
  <div class="left-panel" :style="{ backgroundImage: `url(${imageUrl})` }">
    <div class="image-wrapper">
      <slot></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ imageUrl: string }>()
</script>

<style scoped>
/* 左侧面板样式 */
.left-panel {
  width: 60vw;                  /* 占据视口宽度的 60% */
  height: 100vh;                /* 高度为整个视口高度 */
  background-size: cover;      /* 背景图铺满，不拉伸变形 */
  background-position: center; /* 背景图居中显示 */
}

/* 遮罩 + 内容容器 */
.image-wrapper {
  width: 100%;
  height: 100%;
  background-color: rgba(185, 175, 187, 0.441); /* 添加半透明遮罩层（灰紫色） */
  display: flex;
  justify-content: center;      /* 水平居中插槽内容 */
  align-items: center;          /* 垂直居中插槽内容 */
  flex-direction: column;       /* 垂直排列插槽中的元素 */
}
</style>

