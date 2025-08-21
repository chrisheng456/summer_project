
import type { MeetingDetail, AgendaItem } from '@/types'               
import type { MeetingData, MeetingRecord } from '@/types/interface'    


export function mapMeetingDetailToMeetingData(detail?: MeetingDetail | null): MeetingData {
  console.group('[transform.mapMeetingDetailToMeetingData]')
  console.log('detail exists?', !!detail)
  console.log('agenda isArray?', Array.isArray(detail?.agenda), detail?.agenda)

  if (!detail) {
    return { abstract: '', meetings: [] }
  }


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


  const meetings: MeetingRecord[] = (detail.agenda ?? []).map(a =>
    mapAgendaItemToRecord(a, detail)
  )

  console.log('abstract =', abstract)
  console.log('meetings.length =', meetings.length)
  console.groupEnd()

  return { abstract, meetings }
}


/** Single agenda item -> MeetingRecord */
function mapAgendaItemToRecord(item: AgendaItem, detail: MeetingDetail): MeetingRecord {
  const label = (item.label ?? '').toString().toLowerCase().trim()

  const actions: string[]   = label === 'action'
    ? [item.explanation?.trim() ?? ''] 
    : []

  const decisions: string[] = label.includes('decision')
    ? [item.explanation?.trim() ?? '']
    : []

  const conflicts: string[] = label === 'conflict'
    ? [item.explanation?.trim() ?? '']
    : []

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
