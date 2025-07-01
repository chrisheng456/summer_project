<template>
  <div id="app">
    <!-- RouterView 永远展示 -->
    <RouterView></RouterView>

  </div>
</template>

<script lang ="ts" setup name ="App">//vue3写法，组件名写在<script>里面

import { RouterView} from 'vue-router';

</script> 

<!-- 加入局部样式，使得css作用域是本vue的html -->
<style scoped>

</style>