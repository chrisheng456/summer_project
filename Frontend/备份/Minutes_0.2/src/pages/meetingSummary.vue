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
          <p>（这里为空）</p>
          <h2>Transcript</h2>
          <p>Here will be the transcript content when Summary is active.</p>
        </div>
        <!-- Details内容区 -->
        <div v-show="activeTab === 'details'" id="detailsContent">
          <h2>Details</h2>
          <p>（这里为空）</p>
        </div>
      </div>
    </div>
  </template>
  
  <script lang ="ts" setup name = "meetingSummary">
  import { ref } from 'vue';
  import { RouterView,useRouter} from 'vue-router';

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
    font-family: Arial, sans-serif;
    background-color: #f5f5f5;
    margin: 0;
    padding: 20px;
  }
  
  a {
    color: purple;
    text-decoration: none;
    font-size: 14px;
  }
  
  .back-link {
    display: inline-block;
    margin-bottom: 20px;
  }
  
  /* 标题样式 */
  h1 {
    font-weight: bold;
    font-size: 24px;
  }
  
  .date {
    margin-left: 10px;
    font-size: 16px;
    color: #555;
  }
  
  /* 主题区 */
  .section {
    background-color: #fff;
    border-radius: 4px;
    padding: 10px;
    margin-top: 20px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }
  
  /* 标签组按钮样式 */
  .tabs {
    display: flex;
    margin-bottom: 10px;
  }
  
  .tab {
    padding: 8px 16px;
    border: none;
    background-color: #eee;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.3s;
    margin-right: 4px;
    border-radius: 4px 4px 0 0;
  }
  
  .tab.active {
    background-color: #fff;
    border-bottom: 2px solid #fff;
    font-weight: bold;
  }
  
  /* 内容区域 */
  </style>
  