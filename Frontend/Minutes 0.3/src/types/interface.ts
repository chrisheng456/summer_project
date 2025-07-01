export interface MeetingRecord {
  title: string;               // 会议标题
  attendess: string;           // 参会人员（逗号分隔字符串）
  id: string | number;         // 会议 ID，可为字符串或数字
  time: string;                // 会议时间（字符串格式，如 "2025-07-01 10:00"）
  speaker: string;             // 发言人
  actions: string[];           // 行动项列表
  decisions: string[];         // 决策内容列表
  conflicts: string[];         // 冲突信息列表
  summary: string;             // 会议总结
}

