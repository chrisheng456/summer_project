import loadData from './loadData'
import type { MeetingRecord } from '@/types/interface'

//1.使用异步函数从端口截取信息
//: Promise<MeetingRecord[]>：说明这个函数最终返回一个 MeetingRecord[] 数组的 Promise，也就是说返回值是「会议记录数组」。
export async function fetchMeetingsData(): Promise<MeetingRecord[]> {
  try {
    const res = await loadData('/api/meetings')
  //res?.data 是 可选链语法，防止 res 是 null 或 undefined 时出错。
	//?? [] 表示：如果 res?.data 为 null 或 undefined，就返回空数组。
    return res?.data ?? []
  } catch (error) {
    console.error('Failure:', error)
    return [] // 返回空数组兜底，防止页面崩溃
  }
}