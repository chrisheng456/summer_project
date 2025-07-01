// src/mock/mockMeetings.ts
import Mock from 'mockjs';

const data = Mock.mock({
  'meetings|5': [
    {
      'id|+1': 1,
      'title': '@ctitle(5, 10)',
      'attendess': '@cname, @cname, @cname',
      'time': '@datetime',
      'speaker': '@cname',
      'actions|1-5': ['@sentence(6, 12)'],
      'decisions|0-2': ['@csentence(6, 15)'],
      'conflicts|0-3': ['@csentence(5, 10)'],
      'summary': '@paragraph(1, 2)',
    },
  ],
});

// 拦截 GET 请求
Mock.mock('/api/meetings', 'get', () => {
  return {
    code: 200,
    message: 'success',
    data: data.meetings,
  };
});