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
      this.details[key] = detail
      if (!this.analyzedKeys.includes(key)) this.analyzedKeys.push(key)
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