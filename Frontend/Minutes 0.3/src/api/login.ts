import axios from 'axios'

export async function login(username: string, password: string) {
  try {
    const res = await axios.post('/api/login', { username, password })
    return res.data
  } catch (error) {
    console.error('Login error:', error)
    throw error
  }
}