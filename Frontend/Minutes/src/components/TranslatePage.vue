<template>
  <div>
    <h1>会议纪要</h1>
    <div v-if="minutes">
      <h3>总文字</h3>
      <p>{{ minutes.transcription }}</p >

      <h3>逐句内容</h3>
      <ul>
        <li v-for="line in minutes.lines" :key="line.offset">
          <strong>{{ line.speaker }}:</strong> {{ line.text }}
        </li>
      </ul>
    </div>
    <div v-else>加载中...</div>

    <hr />

    <h3>会议 Word 文档</h3>
    <a href=" " download>下载 Word 文件</a >
  </div>
</template>

<script>
export default {
  data() {
    return {
      minutes: null
    };
  },
  mounted() {
    fetch('/meeting_minutes.json')
      .then(res => res.json())
      .then(data => {
        this.minutes = data;
      });
  }
};
</script>

<style>
body {
  font-family: sans-serif;
}
</style>