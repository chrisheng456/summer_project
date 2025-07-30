import loadData from './loadData'
import type { MeetingData } from '@/types/interface'

//1.使用异步函数从端口截取信息
//: Promise<MeetingRecord[]>：说明这个函数最终返回一个 MeetingRecord[] 数组的 Promise，也就是说返回值是「会议记录数组」。
export async function fetchMeetingsData(): Promise<MeetingData> {
  try {
    const res = await loadData('/api/meetings')
    return res?.data ?? { abstract: '', meetings: [] }
  } catch (error) {
    console.error('Failure:', error)
    return { abstract: '', meetings: [] } // 返回空数组兜底，防止页面崩溃
  }
}

