<template>
  <div class="grid">
    <el-card class="card">
      <h3>Tasks</h3>
      <p><strong>Task:</strong> Follow up with MTI...</p>
      <p><strong>Owner:</strong> Counselor Miller</p>
      <p><strong>Due Date:</strong> 2024-10-10</p>
    </el-card>

    <el-card class="card">
      <h3>Decisions</h3>
      <p><strong>Decision:</strong> Approved...</p>
      <p><strong>Date:</strong> 2024-10-03T00:00:00.000Z</p>
    </el-card>

    <el-card class="card">
      <h3>Questions</h3>
      <p><strong>Question:</strong> Is there a plan...</p>
      <p><strong>Status:</strong> answered</p>
    </el-card>

    <el-card class="card">
      <h3>Insights</h3>
      <p><strong>Insight:</strong> Community members...</p>
    </el-card>
  </div>
</template>

<script setup lang="ts">
// 需先安装并全局注册 Element Plus
</script>

<style scoped>
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.card {
  padding: 10px;
}
</style>