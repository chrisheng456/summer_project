import { defineStore } from 'pinia'
import type { MeetingDetail } from '@/types' // 你的类型定义里导出的 MeetingDetail

type Key = string // 形如 `${schemeId}:${meetingId}`

export const useMeetingStore = defineStore('meeting', {
  state: () => ({
    /** 每场会议的原始 analyze 结果，key = `${schemeId}:${meetingId}` */
    details: {} as Record<Key, MeetingDetail>,
    /** 已完成分析的 key 列表（可用于控制“View Details”是否可点） */
    analyzedKeys: [] as Key[],
  }),

  getters: {
    /** 通过 key 取结果 */
    getByKey: (state) => (key: Key) => state.details[key],
    /** 通过两个 id 取结果 */
    getByIds: (state) => (schemeId: string | number, meetingId: string | number) =>
      state.details[`${schemeId}:${meetingId}`],

    /** 是否已完成分析 */
    isAnalyzed: (state) => (schemeId: string | number, meetingId: string | number) =>
      state.analyzedKeys.includes(`${schemeId}:${meetingId}`),
  },

  actions: {
    /** 写入/覆盖一场会议的 analyze 结果 */
    setDetail(detail: MeetingDetail, schemeId: string | number, meetingId: string | number) {
      const key: Key = `${schemeId}:${meetingId}`

      // ✅ 从后端总结果里“捞出”真正的议程，并统一归一化为数组
      const raw: any = detail
      const extractedAgenda =
        raw?.agenda
        ?? raw?.customer_meeting_detail?.agenda
        ?? raw?.customer_meeting_detail?.agenda_items
        ?? raw?.data_cleaning?.agenda
        ?? raw?.data_cleaning?.agenda_items
        ?? []

      const safeDetail: MeetingDetail = {
        ...detail,
        agenda: Array.isArray(extractedAgenda) ? extractedAgenda : [],
      }

      this.details[key] = safeDetail
      if (!this.analyzedKeys.includes(key)) this.analyzedKeys.push(key)

      // ===== 调试输出 =====
      console.group('[meetingStore.setDetail]')
      console.log('key =', key)
      console.log('detail.id =', (detail as any)?.id)
      console.log('agenda extracted from =',
        raw?.agenda ? 'detail.agenda'
        : raw?.customer_meeting_detail?.agenda ? 'customer_meeting_detail.agenda'
        : raw?.customer_meeting_detail?.agenda_items ? 'customer_meeting_detail.agenda_items'
        : raw?.data_cleaning?.agenda ? 'data_cleaning.agenda'
        : raw?.data_cleaning?.agenda_items ? 'data_cleaning.agenda_items'
        : 'none'
      )
      console.log('agenda isArray?', Array.isArray(safeDetail.agenda), 'length =', (safeDetail.agenda as any[]).length)
      console.log('details keys now =', Object.keys(this.details))
      console.groupEnd()
    },

    // ✅ 新增：方法风格
    listAll(): MeetingDetail[] {
      return Object.values(this.details)
    },

    /** 删除一场会议的结果 */
    removeDetail(schemeId: string | number, meetingId: string | number) {
      const key: Key = `${schemeId}:${meetingId}`
      delete this.details[key]
      this.analyzedKeys = this.analyzedKeys.filter(k => k !== key)
    },

    /** 清空 */
    clearAll() {
      this.details = {}
      this.analyzedKeys = []
    },
  },

  // ✅ 如果你装了 pinia-plugin-persistedstate，则打开持久化，刷新后也在
  // persist: { paths: ['details', 'analyzedKeys'] },
})
