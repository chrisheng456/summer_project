// 每一节会议内容
export interface MeetingRecord {
  title: string;
  id: string | number;
  time: string;
  speaker: string;
  actions: string[];
  decisions: string[];
  conflicts: string[];
  summary: string;
}

// 整个会议结构（包含多个 section 和一个 abstract）
export interface MeetingData {
  abstract: string;
  meetings: MeetingRecord[];
}