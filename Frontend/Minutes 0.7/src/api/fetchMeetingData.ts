// src/api/fetchMeetingData.ts

// 引入“底层接口封装”：负责直接请求后端
import { meetingApi } from './modules/meeting'

// 引入“数据整形函数”：把后端的 result_json 映射成页面需要的 MeetingData 形状
import { mapApiResultToMeetingData } from './transform/meeting'

// 引入类型：返回值会是 { abstract, meetings[] }
import type { MeetingData, RawResultJson } from '@/types/interface'

/**
 * 聚合函数：拉取任务列表 → 取已完成任务的结果 → 映射为 MeetingData → 合并返回
 * 页面直接调用这个函数，就能拿到可以渲染的会议数据
 */
export async function fetchMeetingsData(): Promise<MeetingData> {
  try {
    // 1) 向后端要“任务列表” (/tasks)，里面包含每个转写任务的状态
    const tasks = await meetingApi.listTasks()

    // 2) 只挑选出状态为 done 的任务（说明已经处理完成、有结果可取）
    //    (tasks ?? [])：防御写法，tasks 为 null/undefined 时退回空数组，避免报错
    const doneTasks = (tasks ?? []).filter((t: any) => t.status === 'done')

    // 3) 并发拉取每个 done 任务的“结果详情” (/result/{id})
    //    然后把 result_json 映射成 MeetingData（abstract + meetings[]）
    const parts: MeetingData[] = await Promise.all(
      doneTasks.map(async (t: any) => {
        // 任务 id 可能叫 id 或 task_id，这里二选一
        const detail = await meetingApi.getResult(t.id ?? t.task_id)
        // detail.result_json 可能是字符串化 JSON，把它转成 MeetingData 形状
        return mapApiResultToMeetingData((detail?.result_json ?? '{}') as RawResultJson)
      })
    )

    // 4) 如果有多个任务的结果，把它们合并成一个 MeetingData
    //    - abstract：优先取到第一个不为空的摘要
    //    - meetings：把各部分的 meetings 拼接成一个大数组
    const merged: MeetingData = {
      abstract: parts.find(p => p.abstract && p.abstract.trim())?.abstract ?? '',
      meetings: parts.flatMap(p => p.meetings ?? []),
    }

    // 5) 把整理好的数据返回给页面
    return merged
  } catch (e) {
    // 6) 兜底：任何一步失败，打印错误并返回一个“空壳”，避免页面崩
    console.error('fetchMeetingsData failed:', e)
    return { abstract: '', meetings: [] }
  }
}