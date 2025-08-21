import { defineStore } from 'pinia'
import type { MeetingDetail } from '@/types' 

type Key = string 

export const useMeetingStore = defineStore('meeting', {
  state: () => ({
    /** Original analyze results of each meeting, key = `${schemeId}:${meetingId}` */
    details: {} as Record<Key, MeetingDetail>,
    /** List of analyzed keys (used to control whether "View Details" is clickable) */
    analyzedKeys: [] as Key[],
  }),

  getters: {
    /** Get result by key */
    getByKey: (state) => (key: Key) => state.details[key],
    /** Get result by two IDs */
    getByIds: (state) => (schemeId: string | number, meetingId: string | number) =>
      state.details[`${schemeId}:${meetingId}`],

    /** Check if analysis is completed */
    isAnalyzed: (state) => (schemeId: string | number, meetingId: string | number) =>
      state.analyzedKeys.includes(`${schemeId}:${meetingId}`),
  },

  actions: {
    setDetail(detail: MeetingDetail, schemeId: string | number, meetingId: string | number) {
      const key: Key = `${schemeId}:${meetingId}`

      //  Extract "agenda" from backend response and normalize it to an array
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

   
    listAll(): MeetingDetail[] {
      return Object.values(this.details)
    },

    /** Remove result of a meeting */
    removeDetail(schemeId: string | number, meetingId: string | number) {
      const key: Key = `${schemeId}:${meetingId}`
      delete this.details[key]
      this.analyzedKeys = this.analyzedKeys.filter(k => k !== key)
    },

    /** Clear all results */
    clearAll() {
      this.details = {}
      this.analyzedKeys = []
    },
  },


})
