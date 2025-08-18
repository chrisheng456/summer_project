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

   // 3.导出 PDF
  async exportPdf(schemeId: string, meetingId: string): Promise<Blob> {
    const resp = await http.post('/export/pdf', {}, {
      params: { scheme_id: schemeId, meeting_id: meetingId },
      responseType: 'blob',
    })
    return resp.data
  },

  // 4.导出 Word（docx）
  async exportDocx(schemeId: string, meetingId: string): Promise<Blob> {
    // 如果后端是 /export/docx 或其他路径，请替换
    const resp = await http.post('/export/docx', {}, {
      params: { scheme_id: schemeId, meeting_id: meetingId },
      responseType: 'blob',
    })
    return resp.data
  },
}