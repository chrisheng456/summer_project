
import { useMeetingStore } from '@/stores/meeting'
import { mapMeetingDetailToMeetingData } from '@/api/transform/meeting'
import type { MeetingData, MeetingDetail } from '@/types'
import type { MeetingRecord } from '@/types/interface'

/**
 * Get render data for a single meeting (read from Pinia + trim)
 */
export async function fetchMeetingsData(
  schemeId: string | number,
  meetingId: string | number
): Promise<MeetingData> {
  const store = useMeetingStore()

  // ===== Debug: start =====
  console.group('[fetchMeetingsData]')
  console.log('input schemeId=', schemeId, 'meetingId=', meetingId)
  const asNumSid = Number(schemeId)
  const asNumMid = Number(meetingId)
  const asStrSid = String(schemeId)
  const asStrMid = String(meetingId)
  console.log('as number ids =', asNumSid, asNumMid, 'as string ids =', asStrSid, asStrMid)

  const keys = Object.keys(store.details ?? {})
  console.log('store.details keys =', keys)

  // Try both number & string keys to avoid key-type mismatch
  const byNum  = store.getByIds?.(asNumSid, asNumMid) as MeetingDetail | undefined
  const byStr  = store.getByIds?.(asStrSid, asStrMid) as MeetingDetail | undefined
  const detail = (byNum ?? byStr) ?? null

  console.log('matched by number?', !!byNum, 'matched by string?', !!byStr)
  if (detail) {
    console.log('detail found ✅')
    console.log('agenda isArray?', Array.isArray((detail as any)?.agenda))
    if (Array.isArray((detail as any)?.agenda)) {
      console.log('agenda length =', (detail as any).agenda.length)
    } else {
      console.warn('agenda is NOT an array ❗️ raw agenda =', (detail as any)?.agenda)
    }
  } else {
    console.warn('detail NOT found ❗️')
    console.warn('expected keys maybe:', `${asStrSid}:${asStrMid}`, 'or', `${asNumSid}:${asNumMid}`)
    console.groupEnd()
    return { abstract: '', meetings: [] }
  }
  // ===== Debug: end (detail found) =====

  // Map to the structure required by the page
  const result = mapMeetingDetailToMeetingData(detail)

  // ===== Debug transform result =====
  console.log('transform result.abstract length =', (result.abstract ?? '').length)
  console.log('transform result.meetings length =', (result.meetings ?? []).length)
  if (result.meetings?.length) {
    const first = result.meetings[0]
    console.log('first meeting sample =', {
      title: first?.title,
      id: first?.id,
      time: first?.time,
      hasActions: !!first?.actions?.length,
      hasDecisions: !!first?.decisions?.length,
      hasConflicts: !!first?.conflicts?.length,
      summaryLen: (first?.summary ?? '').length,
    })
  }
  console.groupEnd()
  // ===== Debug: end =====

  return result
}
