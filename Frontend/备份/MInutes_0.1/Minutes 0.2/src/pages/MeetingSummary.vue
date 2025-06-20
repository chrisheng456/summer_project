<template>
    <div>
      <!-- 返回按钮 -->
      <a href="#" class="back-link">Back to Dashboard</a>
  
      <!-- 会议日期和标题 -->
      <h1>会议标题</h1>
      <span class="date">{{ currentDate }}</span>
  
      <!-- Summary部分 -->
      <div class="section" id="summarySection">
        <div class="tabs">

          <button class="tab" :class="{ active: activeTab === 'summary' }"@click="setTab('summary')" >
            Summary
          </button>

          <button class="tab" :class="{ active: activeTab === 'details' }"@click="setTab('details')">
            Details
          </button>

        </div>
        <!-- Summary内容区 -->
        <div v-show="activeTab === 'summary'" id="summaryContent">
          <h2>Summary</h2>

          <!-- 这里放嵌套 -->
          <p>（这里为空）</p>
          <h2>Transcript</h2>
          <p>Here will be the transcript content when Summary is active.</p>
        </div>
        <!-- Details内容区 -->
        <div v-show="activeTab === 'details'" id="detailsContent">
          <h2>Details</h2>
          <analysisPage></analysisPage>
        </div>
      </div>
    </div>
  </template>

  
  <script lang ="ts" setup name = "meetingSummary">
  import { ref } from 'vue';
  import { RouterView,useRouter} from 'vue-router';
import AnalysisPage from './AnalysisPage.vue';
 

  // 当前日期，可以根据需要设置格式
  const currentDate = ref(new Date().toLocaleDateString('zh-CN'));
  
  // 当前激活的标签
  const activeTab = ref('summary');
  
  // 方法切换标签
  function setTab(tab: string) {
    activeTab.value = tab;
  }
  </script>
  

  <style scoped>
  body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #f9f9f9;
  margin: 0;
  padding: 20px;
  color: #333;
  line-height: 1.6;
}

a {
  color: #6a0dad;
  text-decoration: none;
  font-size: 14px;
  transition: color 0.2s;
}

a:hover {
  color: #4b0082;
  text-decoration: underline;
}

.back-link {
  display: inline-block;
  margin-bottom: 20px;
  font-weight: 600;
}

h1 {
  font-size: 32px;
  margin-bottom: 12px;
  color: #222;
  text-align: center;
}

.date {
  display: block;
  text-align: center;
  font-size: 16px;
  color: #777;
  margin-bottom: 20px;
}

.section {
  background-color: #fff;
  border-radius: 10px;
  padding: 24px;
  margin-top: 30px;
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.08);
  transition: box-shadow 0.3s ease;
}

.section:hover {
  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.12);
}

.tabs {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  flex-wrap: wrap;
  justify-content: center;
}

.tab {
  padding: 10px 20px;
  border: none;
  background-color: #e0e0e0;
  cursor: pointer;
  font-size: 15px;
  border-radius: 6px 6px 0 0;
  transition: background-color 0.3s, transform 0.2s;
}

.tab:hover {
  background-color: #d5d5d5;
  transform: translateY(-1px);
}

.tab.active {
  background-color: #ffffff;
  border-bottom: 2px solid #ffffff;
  font-weight: bold;
  box-shadow: 0 -2px 4px rgba(0, 0, 0, 0.05);
}

h2 {
  font-size: 20px;
  margin-top: 16px;
  color: #444;
}

p {
  margin: 8px 0;
  color: #555;
}
  /* 内容区域 */
  </style>
  