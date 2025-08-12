import Mock from 'mockjs'

const rawMeetings = Mock.mock({
  'meetings|1-7': [
    {
      'id|+1': 1,
      'attendess': '@name, @name, @name',
      'time': '@datetime',
      'speaker': '@name',
      'actions|1-5': ['@sentence(6, 12)'],
      'decisions|0-2': ['@sentence(6, 15)'],
      'conflicts|0-3': ['@sentence(5, 10)'],
      'summary': '@paragraph(1, 2)',
      'duration':''
    }
  ]
})

// 添加统一的 abstract 字段（模拟整场会议的总览摘要）
const data = {
  abstract: Mock.mock('@paragraph(2, 3)'),
  meetings: rawMeetings.meetings.map((item: any) => ({
    ...item,
    title: `Section ${item.id}`
  }))
}

export default data

Mock.mock('/api/meetings', 'get', () => {
  return {
    code: 200,
    message: 'success',
    data: data,
  }
})