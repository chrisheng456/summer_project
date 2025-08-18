 // 通用：后端返回的 ISO 日期时间字符串
export type ISODateTimeString = string;
// 1.发送scheme_id和meetin_id的结果
export interface MeetingDetail {
  id: number;
  name: string;
  /** 会议日期（如 "2024-11-18T00:00:00"） */
  date: ISODateTimeString;
  /** 会议开始时间（如 "2025-06-04T10:00:00"） */
  startTime: ISODateTimeString;
  location: string;

  attendees: Attendee[];
  agenda: AgendaItem[];

  /** 顶层附件，示例为空数组，结构未知先占位 */
  attachment: Attachment[];
}

export interface Attendee {
  id: number;
  name: string;
  attending: boolean;
  userCanEdit: boolean;
}

/** 议程项 */
export interface AgendaItem {
  id: number;
  /** 议题编号（字符串形式，例如 "4.1"） */
  number: string;
  title: string;
  /** 缩进层级：0 为顶级，>0 为子项 */
  indent: number;

  /** 下面这些字段在部分项里不存在，设为可选 */
  calculatedStartTime?: ISODateTimeString;
  lengthMinutes?: number;
  owner?: string;

  /** 如 "Declaration"、"Discussion" 等 */
  action?: string;
  /** 颜色字符串（如 "0x39c0ed"） */
  action_colour?: string;

  htmlText?: string;

  /** 每个议程项自己的附件，示例为空数组 */
  attachment: Attachment[];

  /** 语音转写的时间轴行，部分议题没有则缺省 */
  lines?: TranscriptLine[];

  /** 模型给的分类标签，可能为 null */
  label?: string | null;
  label_score?: number | null;

  /** 解释/摘要，可能为空字符串 */
  explanation?: string;
  summary?: string;
}

export interface TranscriptLine {
  /** 秒 */
  start: number;
  /** 秒 */
  end: number;
  text: string;
  /** 例如 "SPEAKER_01"、"SPEAKER_00"、"Unknown" */
  speaker: string;
}

/** 附件占位：后端暂未给出结构，先用字典类型 */
export type Attachment = Record<string, unknown>;
