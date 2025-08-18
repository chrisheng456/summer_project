import type { AgendaItem, MeetingDetail } from "@/types"

// 将 analyze 返回的“总结果”适配为 MeetingDetail
export function adaptApiResultToMeetingDetail(resp: any): MeetingDetail {
  // 1) 源数据候选（后端可能把会议信息放在 customer_meeting_detail 或 data_cleaning）
  const src = resp?.customer_meeting_detail ?? resp?.data_cleaning ?? resp ?? {}

  // 2) 会议元信息兜底
  const id = Number(src?.id ?? src?.meeting_id ?? 0)
  const name = String(src?.name ?? src?.title ?? 'Untitled Meeting')
  const date = String(src?.date ?? src?.meeting_date ?? src?.start_time ?? '')
  const startTime = String(src?.startTime ?? src?.start_time ?? src?.start ?? date ?? '')
  const location = String(src?.location ?? '')

  const attendees: any[] = Array.isArray(src?.attendees) ? src.attendees : []

  // 3) 议程候选路径：agenda / agenda_items / sections
  let agendaRaw: any[] =
    (Array.isArray(src?.agenda) && src.agenda) ||
    (Array.isArray(src?.agenda_items) && src.agenda_items) ||
    (Array.isArray(src?.sections) && src.sections) ||
    []

  // 如果是 sections 这种 {title, summary, items[]}，扁平化成 items；没有 items 就把 section 自身当作一条
  if (Array.isArray(src?.sections) && src.sections.length) {
    agendaRaw = src.sections.flatMap((s: any, idx: number) =>
      Array.isArray(s?.items) && s.items.length
        ? s.items
        : [{ id: idx + 1, number: String(idx + 1), title: s?.title, summary: s?.summary }]
    )
  }

  // 4) 若以上都没有，但有转写，则用 paragraphs 兜底生成“伪议程”
  if ((!agendaRaw || agendaRaw.length === 0) && Array.isArray(resp?.speech_to_text?.paragraphs)) {
    agendaRaw = resp.speech_to_text.paragraphs.map((p: any, i: number) => ({
      id: i + 1,
      number: String(i + 1),
      title: p?.speaker ? `Speaker ${p.speaker}` : `Paragraph ${i + 1}`,
      summary: p?.text ?? '',
      calculatedStartTime: p?.start ?? '',
      label: 'transcript',
    }))
  }

  // 5) 规范化为 AgendaItem[]
  const agenda: AgendaItem[] = (agendaRaw ?? []).map((a: any, i: number) => ({
    id: Number(a?.id ?? i + 1),
    number: String(a?.number ?? i + 1),
    title: String(a?.title ?? a?.name ?? `Item ${i + 1}`),
    indent: Number(a?.indent ?? 0),
    calculatedStartTime: a?.calculatedStartTime ?? a?.start_time ?? a?.start ?? undefined,
    lengthMinutes: a?.lengthMinutes ?? a?.length_minutes ?? undefined,
    owner: a?.owner ?? undefined,
    action: a?.action ?? undefined,
    action_colour: a?.action_colour ?? undefined,
    htmlText: a?.htmlText ?? a?.html ?? undefined,
    attachment: Array.isArray(a?.attachment) ? a.attachment : [],
    lines: Array.isArray(a?.lines) ? a.lines : undefined,
    label: a?.label ?? null,
    label_score: a?.label_score ?? null,
    explanation: a?.explanation ?? '',
    summary: a?.summary ?? '',
  }))

  return {
    id,
    name,
    date,
    startTime,
    location,
    attendees,
    agenda,
    attachment: Array.isArray(src?.attachment) ? src.attachment : [],
  }
}
