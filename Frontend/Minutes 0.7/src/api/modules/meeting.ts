import { http } from '../http'
import type { ConvertResp, ResultDetail, TaskItem } from '@/types/interface' // 假设这些类型你已在别处定义；也可放在本文件
/**
 * ====== API 封装 ======
 * 统一通过 http(axios 实例) 调用，baseURL/超时/Token 拦截器等在 http.ts 配置
 */

export const meetingApi = {
  /**
   * 上传文件启动转换（POST /convert）
   * 使用 multipart/form-data 提交；通常返回任务 id，用于后续 /result/{id} 轮询
   */
  async convert(file: File | Blob, extra: Record<string, any> = {}): Promise<ConvertResp> {
    const fd = new FormData()                                // 新建一个 FormData，用于“带文件的表单”提交
    const fieldName = 'file'                                 // 与后端约定的字段名（通常叫 'file'）
    fd.append(fieldName, file)                               // 把文件塞进表单：键是 'file'，值是 File/Blob

    for (const [k, v] of Object.entries(extra)) {            // 把额外参数一并追加进表单（比如 language、meetingId 等）
      fd.append(k, String(v))                                // FormData 的值必须是字符串或 Blob，这里统一转字符串
    }

    const resp = await http.post('/convert', fd, {           // 用 axios 实例发 POST 请求到 /convert
      headers: { 'Content-Type': 'multipart/form-data' },    // 指定是“文件表单”；axios 会自动加 boundary
    })                                                       // 等待请求完成，拿到响应对象

    return resp.data as ConvertResp                          // 只返回业务数据 data，并断言为 ConvertResp 类型
  },

  /**
   * 携带 schemeId / meetingId 的上传（POST /convert/{schemeId}/{meetingId}）
   * 当需要与某个方案/会议绑定时使用这个接口
   */
  async convertWithId(
    schemeId: string | number,                               // 方案 ID（路径参数）
    meetingId: string | number,                              // 会议 ID（路径参数）
    file: File | Blob                                        // 待上传的媒体文件
  ): Promise<ConvertResp> {
    const fd = new FormData()                                // 初始化 FormData
    fd.append('file', file)                                  // 同样以 'file' 为键追加文件

    const url = `/convert/${encodeURIComponent(String(schemeId))}/${encodeURIComponent(String(meetingId))}`
                                                              // 组装路径并做 URL 编码，避免特殊字符导致 404
    const resp = await http.post(url, fd, {                  // 发起 POST 到拼好的路径
      headers: { 'Content-Type': 'multipart/form-data' },    // 指定为 multipart/form-data
    })                                                       // 等待响应

    return resp.data as ConvertResp                          // 返回 data，断言为 ConvertResp
  },

  /**
   * 查询任务结果（GET /result/{task_id}）
   * 前端通常“轮询”这个接口直到 status === 'done'
   */
  async getResult(taskId: string): Promise<ResultDetail> {
    const url = `/result/${encodeURIComponent(taskId)}`      // 拼接路径并 URL 编码 taskId
    const resp = await http.get(url)                         // GET 请求查询任务结果
    return resp.data as ResultDetail                         // 返回 data，断言为 ResultDetail
  },
  /**
   * 获取任务列表（GET /tasks）
   * 可用于“上传历史”或“处理队列”页面
   */
  async listTasks(): Promise<TaskItem[]> {
    const resp = await http.get('/tasks')                    // 调用 /tasks 获取任务数组
    return resp.data as TaskItem[]                           // 返回 data，断言为 TaskItem[]
  },

  /**
   * 导出 PDF（GET /export/pdf/{task_id}）
   * 返回 Blob（二进制）；前端自行触发下载
   */
  async exportPdf(taskId: string): Promise<Blob> {
    const url = `/export/pdf/${encodeURIComponent(taskId)}`  // 拼接导出 PDF 的路径
    const resp = await http.get(url, {                       // 发起 GET 请求
      responseType: 'blob',                                  // 关键：告诉 axios 以 Blob 接收二进制
    })                                                       // 等待响应
    return resp.data as Blob                                 // 返回 Blob 数据
  },

  /**
   * 导出 Word（GET /export/word/{task_id}）
   * 返回 Blob（二进制）；前端自行触发下载
   */
  async exportWord(taskId: string): Promise<Blob> {
    const url = `/export/word/${encodeURIComponent(taskId)}` // 拼接导出 Word 的路径
    const resp = await http.get(url, {                       // 发起 GET 请求
      responseType: 'blob',                                  // 以 Blob 接收
    })                                                       // 等待响应
    return resp.data as Blob                                 // 返回 Blob 数据
  },
}