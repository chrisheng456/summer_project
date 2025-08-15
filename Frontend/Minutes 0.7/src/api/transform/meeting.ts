// src/api/transform/meeting.ts

// 从类型文件里拿到要用到的类型
import type { MeetingData, MeetingRecord, RawResultJson } from '@/types/interface'

/**
 * 把 /result 接口里的 result_json（可能是字符串，也可能是对象）
 * 统一“整理/修剪”为页面需要的 MeetingData 形状
 */
export function mapApiResultToMeetingData(result_json: RawResultJson): MeetingData {
  // 如果是字符串，先安全地 JSON.parse；否则直接用对象；再兜底一个空对象
  const raw =
    typeof result_json === 'string'
      ? safeParse(result_json)
      : result_json || {}

  // 取出议程列表：没有就给空数组，防止后面 map 报错
  const agenda: any[] = Array.isArray(raw?.agenda) ? raw.agenda : []

  // 把每一条议程，映射成页面用的 MeetingRecord
  const meetings: MeetingRecord[] = agenda.map((item, idx): MeetingRecord => {
    // 从行文本 + 模型处理结果里，收集出一堆“句子”
    const sentences = gatherSentences(item)

    // 用很简单的关键词规则，分出“行动项 / 决议”
    const [actions, decisions] = splitActionsDecisions(sentences)

    // 返回一条 MeetingRecord（页面上的一个 Section）
    return {
      title: formatTitle(item, idx),                      // 标题：“编号. 标题名”，没标题就兜底
      id: item?.id ?? item?.number ?? String(idx + 1),   // id：优先后端给的，没有就用序号
      time: item?.calculatedStartTime ?? raw?.startTime ?? '', // 开始时间：优先取该条的，缺了用整场的
      speaker: item?.owner ?? '',                        // 负责人/发言人
      actions,                                           // 行动项（数组）
      decisions,                                         // 决议（数组）
      conflicts: [],                                     // 目前后端没给冲突信息，这里先留空数组
      summary:
        item?.summary?.trim?.() ||                       // 优先用后端给的 summary
        sentences.slice(0, 2).join(' ') ||               // 没有的话，取前两句拼成一个“简易摘要”
        ''                                               // 还没有就给空字符串
    }
  })

  // 计算整场会议的“摘要”：
  // 先找第一个议程自带的 summary；没找到就把各 section 的 summary 取前两条拼一下；还没有就空
  const abstract =
    (agenda.find(a => a?.summary)?.summary as string) ||
    meetings
      .map(m => m.summary)
      .filter(Boolean)
      .slice(0, 2)
      .join('  ') ||
    ''

  // 返回给页面的最终结构
  return { abstract, meetings }
}

/** 安全 JSON.parse：失败就返回 {}，避免异常把页面卡死 */
function safeParse(text: string): any {
  try {
    return JSON.parse(text)
  } catch {
    return {}
  }
}

/**
 * 从一条议程 item 中，抽取所有“可读句子”
 * 来源：lines[].text + lines[].processed[].sentence
 */
function gatherSentences(item: any): string[] {
  const lines = Array.isArray(item?.lines) ? item.lines : [] // 取出行列表
  const out: string[] = []                                   // 临时存放句子的数组

  for (const ln of lines) {
    // 原始识别文本（有就收集）
    if (typeof ln?.text === 'string' && ln.text.trim()) {
      out.push(ln.text.trim())
    }
    // 模型处理后的句子（有就收集）
    if (Array.isArray(ln?.processed)) {
      for (const p of ln.processed) {
        if (p?.sentence) out.push(String(p.sentence).trim())
      }
    }
  }
  // 过滤空值 + 去重，返回
  return dedupe(out.filter(Boolean))
}

/**
 * 超简规则：把句子按关键词切成“行动项 / 决议”
 * 你可以随时替换为更智能的规则或直接用后端字段
 */
function splitActionsDecisions(sentences: string[]): [string[], string[]] {
  const actions: string[] = []
  const decisions: string[] = []

  // 判断“决议”的关键词
  const decisionRe = /\b(approved|agreed|decided|resolved|concluded|confirmed)\b/i
  // 判断“行动”的关键词
  const actionRe =
    /\b(approve|assign|action|please|we will|we'll|to|ensure|prepare|follow up|schedule|update|send|share|review|investigate|implement|fix)\b/i

  // 逐句判断归类
  for (const s of sentences) {
    if (decisionRe.test(s)) decisions.push(s)
    else if (actionRe.test(s)) actions.push(s)
  }
  // 去重后返回
  return [dedupe(actions), dedupe(decisions)]
}

/** 标题：优先“number. title”，都缺就用“序号. Untitled” */
function formatTitle(item: any, idx: number): string {
  const n = item?.number ?? String(idx + 1) // number 缺了就用序号
  const t = item?.title ?? 'Untitled'       // title 缺了就用 Untitled
  return `${n}. ${t}`
}

/** 小工具：数组去重 */
function dedupe<T>(arr: T[]): T[] {
  return Array.from(new Set(arr))
}