//1.登陆返回json数据

// 请求体
export interface LoginReq {
  username: string;
  password: string;
}

// 会议项
export interface Meetings {
  scheme_id: number;
  scheme_name: string;
  meeting_id: number;
  title: string;
  date: string; // ISO 字符串
}

// 响应体
export interface LoginResp {
  ok: boolean;
  token: string;  
  meetings?: Meetings[];//当前用户下可以没有会议
}