declare module 'mockjs';
import Mock from 'mockjs';

const data = Mock.mock({
  'meetings|5': [ // 生成 5 条会议记录
    {
      'id|+1': 1,
      'title': '@ctitle(5, 10)',           // 中文标题，5-10个字
      'attendess': '@cname, @cname, @cname', // 假参会人名
      'time': '@datetime',                 // 随机日期时间
      'speaker': '@cname',                 // 发言人
      'actions|1-5': ['@sentence(6, 12)'], // 随机1-5条行动项，随机6-12个单词
      'decisions|0-2': ['@csentence(6, 15)'], // 决策内容
      'conflicts|0-3': ['@csentence(5, 10)'], // 可无可有
      'summary': '@paragraph(1, 2)'        // 简要总结
    }
  ]
});


