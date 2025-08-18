import { defineStore } from 'pinia'

export interface User {
  id: number
  name: string
  avatarUrl?: string
  email?: string
}

export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null as User | null,
    token: '' as string
  }),
  actions: {
    login(user: User, token: string) {
      this.user = user
      this.token = token
      // 可选：持久化
      localStorage.setItem('auth_user', JSON.stringify(user))
      localStorage.setItem('auth_token', token)
    },
    loadFromStorage() {
      try {
        const u = localStorage.getItem('auth_user')
        const t = localStorage.getItem('auth_token')
        this.user = u ? JSON.parse(u) : null
        this.token = t || ''
      } catch {
        this.user = null
        this.token = ''
      }
    },
    logout() {
      this.user = null
      this.token = ''
      localStorage.removeItem('auth_user')
      localStorage.removeItem('auth_token')
    }
  }
})