<template>
  <div class="left-panel" :style="backgroundStyle">
    <div class="image-wrapper">
      <slot></slot>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// Receive props
const props = defineProps<{ imageUrl: string }>()

// Dynamic background style, prevent incorrect image path handling
const backgroundStyle = computed(() => ({
  backgroundImage: `url(${props.imageUrl})`,
  backgroundSize: 'cover',
  backgroundPosition: 'center',
}))
</script>

<style scoped>
.left-panel {
  width: 55vw;
  height: 100vh;
}

/* Overlay + content container */
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
