// src/api/transform/meeting.ts
// 把后端 MeetingDetail → MeetingData（abstract + meetings[]）

import type { MeetingDetail, AgendaItem } from '@/types'               // 后端返回类型
import type { MeetingData, MeetingRecord } from '@/types/interface'    // 目标前端类型

/** 主函数：MeetingDetail -> MeetingData */
export function mapMeetingDetailToMeetingData(detail?: MeetingDetail | null): MeetingData {
  console.group('[transform.mapMeetingDetailToMeetingData]')
  console.log('detail exists?', !!detail)
  console.log('agenda isArray?', Array.isArray(detail?.agenda), detail?.agenda)

  if (!detail) {
    return { abstract: '', meetings: [] }
  }

  // ============= 修改部分开始：竖排拼接 abstract（英文标签） =============
  let abstractParts: string[] = []

  if (detail.name) {
    abstractParts.push(`Meeting Name: ${detail.name}`)
  }
  if (detail.date) {
    abstractParts.push(`Date: ${detail.date}`)
  }
  if (detail.startTime) {
    abstractParts.push(`Start Time: ${detail.startTime}`)
  }
  if (detail.location) {
    abstractParts.push(`Location: ${detail.location}`)
  }
  if (detail.attendees?.length) {
    const attendeeNames = detail.attendees.map(a => a.name).join(', ')
    abstractParts.push(`Attendees: ${attendeeNames}`)
  }

  const abstract = abstractParts.join('\n')
  // ============= 修改部分结束 =============

  const meetings: MeetingRecord[] = (detail.agenda ?? []).map(a =>
    mapAgendaItemToRecord(a, detail)
  )

  console.log('abstract =', abstract)
  console.log('meetings.length =', meetings.length)
  console.groupEnd()

  return { abstract, meetings }
}

/** 单个议程项 -> MeetingRecord */
function mapAgendaItemToRecord(item: AgendaItem, detail: MeetingDetail): MeetingRecord {
  const label = (item.label ?? '').toString().toLowerCase().trim()

  // ================= 修改部分 =================
  // 根据 label 放入 actions / decisions / conflicts，存 explanation
  const actions: string[]   = label === 'action'
    ? [item.explanation?.trim() ?? ''] 
    : []

  const decisions: string[] = label.includes('decision')
    ? [item.explanation?.trim() ?? '']
    : []

  const conflicts: string[] = label === 'conflict'
    ? [item.explanation?.trim() ?? '']
    : []
  // ================= 修改部分结束 =================

  return {
    title: item.title ?? '',
    id: item.id ?? item.number ?? '',
    time: item.calculatedStartTime ?? detail.startTime ?? detail.date ?? '',
    speaker: '',
    actions,
    decisions,
    conflicts,
    summary: item.summary?.trim() || item.explanation?.trim() || '',
  }
}
