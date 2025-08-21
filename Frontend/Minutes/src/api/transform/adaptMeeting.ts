import type { AgendaItem, MeetingDetail } from "@/types"

// Adapt the "overall result" returned by analyze into MeetingDetail
export function adaptApiResultToMeetingDetail(resp: any): MeetingDetail {
  // 1) Source candidates (backend may put meeting info in customer_meeting_detail or data_cleaning)
  const src = resp?.customer_meeting_detail ?? resp?.data_cleaning ?? resp ?? {}

  // 2) Meeting metadata fallback
  const id = Number(src?.id ?? src?.meeting_id ?? 0)
  const name = String(src?.name ?? src?.title ?? 'Untitled Meeting')
  const date = String(src?.date ?? src?.meeting_date ?? src?.start_time ?? '')
  const startTime = String(src?.startTime ?? src?.start_time ?? src?.start ?? date ?? '')
  const location = String(src?.location ?? '')

  const attendees: any[] = Array.isArray(src?.attendees) ? src.attendees : []

  // 3) Agenda candidate paths: agenda / agenda_items / sections
  let agendaRaw: any[] =
    (Array.isArray(src?.agenda) && src.agenda) ||
    (Array.isArray(src?.agenda_items) && src.agenda_items) ||
    (Array.isArray(src?.sections) && src.sections) ||
    []

  // If it's "sections" (e.g. {title, summary, items[]}), flatten into items;
  // if no items, treat the section itself as an agenda item
  if (Array.isArray(src?.sections) && src.sections.length) {
    agendaRaw = src.sections.flatMap((s: any, idx: number) =>
      Array.isArray(s?.items) && s.items.length
        ? s.items
        : [{ id: idx + 1, number: String(idx + 1), title: s?.title, summary: s?.summary }]
    )
  }

  // 4) If no agenda above but transcript exists, use paragraphs to generate a "pseudo agenda"
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


  // 5) Normalize into AgendaItem[]
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
