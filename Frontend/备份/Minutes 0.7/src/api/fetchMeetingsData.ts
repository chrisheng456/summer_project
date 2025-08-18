// src/api/fetchMeetingData.ts

import { useMeetingStore } from '@/stores/meeting'
import { mapMeetingDetailToMeetingData } from './transform/meeting'
import type { MeetingData, MeetingDetail } from '@/types'             // 后端的 MeetingDetail 类型
import type { MeetingRecord } from '@/types/interface'                 // 目标类型里的子项

/**
 * 获取“单个会议”的渲染数据（从 Pinia 读取 + 修剪）
 * @param schemeId 方案ID（字符串或数字均可）
 * @param meetingId 会议ID（字符串或数字均可）
 * @returns MeetingData（abstract + meetings[]）
 * 
 */
export async function fetchMeetingsData(
  schemeId: string | number,
  meetingId: string | number
): Promise<MeetingData> {
  const store = useMeetingStore()
  const detail = store.getByIds?.(Number(schemeId), Number(meetingId)) as MeetingDetail | null

  if (!detail) {
    // 没有找到就返回空壳，页面不会崩
    return { abstract: '', meetings: [] }
  }

  // 用映射工具修剪为页面所需结构
  return mapMeetingDetailToMeetingData(detail)
}

/**
 * （可选）一次性取“所有已保存会议”的合并结果
 * - abstract：取第一个非空摘要
 * - meetings：把各会议的 meetings 拼起来
 */
export async function fetchAllMeetingsData(): Promise<MeetingData> {
  const store = useMeetingStore()
  const allDetails: MeetingDetail[] = store.listAll?.() ?? []

  if (!allDetails.length) {
    return { abstract: '', meetings: [] }
  }

  const parts = allDetails.map(d => mapMeetingDetailToMeetingData(d))

  const abstract =
    parts.find(p => p.abstract && p.abstract.trim())?.abstract ?? ''

  const meetings: MeetingRecord[] = parts.flatMap(p => p.meetings ?? [])

  return { abstract, meetings }
}