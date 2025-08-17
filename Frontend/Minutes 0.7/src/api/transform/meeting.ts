// src/api/transform/meeting.ts
// 把后端 MeetingDetail → MeetingData（abstract + meetings[]）

import type { MeetingDetail, AgendaItem } from '@/types'               // 后端返回类型
import type { MeetingData, MeetingRecord } from '@/types/interface'    // 目标前端类型

/** 主函数：MeetingDetail -> MeetingData */
export function mapMeetingDetailToMeetingData(detail: MeetingDetail): MeetingData {
  // 1) abstract：优先用第一个有 summary 的议题；没有就空字符串
  const abstract =
    detail.agenda.find(a => (a.summary && a.summary.trim()))?.summary?.trim() ?? ''

  // 2) sections：逐个议程项映射为 MeetingRecord
  const meetings: MeetingRecord[] = (detail.agenda ?? []).map(a =>
    mapAgendaItemToRecord(a, detail)
  )

  return { abstract, meetings }
}

/** 单个议程项 -> MeetingRecord */
function mapAgendaItemToRecord(item: AgendaItem, detail: MeetingDetail): MeetingRecord {
  const label = (item.label ?? '').toString().toLowerCase().trim()

  // 根据 label 放入 actions / decisions / conflicts
  const actions: string[]   = label === 'action'    ? [item.title ?? ''] : []
  const decisions: string[] = label.includes('decision') ? [item.title ?? ''] : []
  const conflicts: string[] = label === 'conflict'  ? [item.title ?? ''] : []

  return {
    title: item.title ?? '',
    id: item.id ?? item.number ?? '',
    time: item.calculatedStartTime ?? detail.startTime ?? detail.date ?? '',
    // 按你的要求：不足部分不要从 lines 里补，speaker 留空即可
    speaker: '',
    actions,
    decisions,
    conflicts,
    // 优先 summary，其次 explanation；都没有就空
    summary: item.summary?.trim() || item.explanation?.trim() || '',
  }
}