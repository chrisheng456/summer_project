import { http } from '../http'
// import type * from '@/types/index';
import type { MeetingDetail } from '@/types' // 或 '@/types'


export const meetingApi = {

// 1.分析页面
  async analyze(file: File, schemeId: string, meetingId: string):Promise<MeetingDetail> {
    const fd = new FormData();
    fd.append('file', file);

    const resp = await http.post('/pipeline/analyze', fd, {
      params: { scheme_id: schemeId, meeting_id: meetingId },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return resp.data;
  },
}