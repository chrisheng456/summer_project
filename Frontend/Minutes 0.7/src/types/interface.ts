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

/** 任务列表中的单条任务结构（可根据后端再细化） */
export interface TaskItem {
  id: string                      
  status: 'processing' |'pending' | 'done' | 'failed'  
  created_at?: string             
  updated_at?: string             
  [k: string]: any                // 允许出现其他任意键
}

/*/convert 返回结构（通常包含 task_id 或 id） */
export interface ConvertResp {
  /** 任务 id，用于后续 /result/{id} 查询 */
  id: string;
  /** 当前状态，后端现在返回了 processing */
  status?: 'processing' | 'pending' | 'done' | 'failed' | string;
  /** 兜底：后端扩展字段 */
  [k: string]: any;
}


/** /result/{task_id} 返回结构 */
// 任务结果详情（/result/{task_id}）
export interface ResultDetail {
  id: string;
  status: 'processing' | 'pending' | 'done' | 'failed';
  // 服务端这里是一个很长的字符串化 JSON
  result_json?: string | MeetingApiResult; 
  error_message?: string | null;
  created_at?: string;
  updated_at?: string;
}
// 把 result_json 里常用字段列出来（可根据真实数据继续补充/收缩）
export interface MeetingApiResult {
  id?: number | string;
  name?: string;
  date?: string;          
  startTime?: string;     
  location?: string;
  attendees?: Array<{ id: number; name: string; attending?: boolean }>;
  agenda?: Array<{
    id?: number | string;
    number?: string;
    title?: string;
    indent?: number;
    calculatedStartTime?: string;
    lengthMinutes?: number;
    owner?: string;
    htmlText?: string;
    summary?: string;
    
    // 语料与分析结果
    lines?: Array<{
      start?: number;
      end?: number;
      text?: string;
      processed?: Array<{
        sentence?: string;
        tokens?: string[];
        pos_tags?: string[];
        lemmas?: string[];
      }>
    }>;
  }>;
  attachment?: any[];
}


export type RawResultJson = string | MeetingApiResult;

