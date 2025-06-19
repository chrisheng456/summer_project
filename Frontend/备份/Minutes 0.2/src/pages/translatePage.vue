<template>
  <div>
    <h1>会议纪要</h1>

    <!-- 如果数据已加载，展示会议内容 -->
    <div v-if="minutes">
      <h3>总文字</h3>
      <p>{{ minutes.transcription }}</p>

      <h3>逐句内容</h3>
      <ul>
        <li v-for="line in minutes.lines" :key="line.offset">
          <strong>{{ line.speaker }}:</strong>
          {{ line.text }}
        </li>
      </ul>
    </div>

    <!-- 如果数据未加载，展示加载中 -->
    <div v-else>加载中...</div>

    <hr />

    <h3>会议 Word 文档</h3>
    <!-- 使用 RouterLink 跳转页面 -->
    <RouterLink to="/downloadPage">下载 Word 文件</RouterLink>
  </div>

  <div @click ="nextPage" class="NextPageButton">Next Page</div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink,useRouter,} from 'vue-router'

const router= useRouter();
const minutes = ref<any>(null);

function nextPage(){
  router.push('/meetingSummary')
}

onMounted(async () => {
  const res = await fetch('/meeting_minutes.json')
  const data = await res.json()
  minutes.value = data
})

</script>

<style>
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #f0f4f8;
  margin: 0;
  padding: 20px;
  color: #333;
}

h1 {
  text-align: center;
  color: #2c3e50;
  margin-bottom: 30px;
}

h3 {
  color: #34495e;
  margin-top: 30px;
}

p {
  line-height: 1.6;
  font-size: 16px;
}

ul {
  list-style-type: none;
  padding-left: 0;
}

li {
  background: #ffffff;
  margin-bottom: 10px;
  padding: 10px 15px;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

strong {
  color: #007BFF;
}

a {
  display: inline-block;
  margin-top: 20px;
  color: #fff;
  background-color: #007BFF;
  padding: 10px 20px;
  border-radius: 5px;
  text-decoration: none;
  transition: background-color 0.3s;
}

a:hover {
  background-color: #0056b3;
}

.NextPageButton {
  margin-top: 40px;
  text-align: center;
  background-color: #28a745;
  color: white;
  padding: 12px 24px;
  font-size: 16px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.NextPageButton:hover {
  background-color: #218838;
}
</style>