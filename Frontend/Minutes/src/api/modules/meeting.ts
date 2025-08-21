import { http } from '../http'
// import type * from '@/types/index';
import type { MeetingDetail } from '@/types' 


export const meetingApi = {

// Analysis page
  async analyze(file: File, schemeId: string, meetingId: string):Promise<MeetingDetail> {
    const fd = new FormData();
    fd.append('file', file);

    const resp = await http.post('/pipeline/analyze', fd, {
      params: { scheme_id: schemeId, meeting_id: meetingId },
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return resp.data;
  },

  // Export PDF
  async exportPdf(schemeId: string, meetingId: string): Promise<Blob> {
    const resp = await http.post('/export/pdf', {}, {
      params: { scheme_id: schemeId, meeting_id: meetingId },
      responseType: 'blob',
    })
    return resp.data
  },

  // Export Word (docx)
  async exportDocx(schemeId: string, meetingId: string): Promise<Blob> {
    const resp = await http.post('/export/docx', {}, {
      params: { scheme_id: schemeId, meeting_id: meetingId },
      responseType: 'blob',
    })
    return resp.data
  },
}