import axios from 'axios';

//创建http实例
export const http = axios.create({
  baseURL: import.meta.env.VITE_BASE_API || 'http://34.105.147.52:8000',
  timeout: 30000,
});




// 带上 token（有登录的话）
http.interceptors.request.use(cfg => {
  const token = localStorage.getItem('token');
  if (token) cfg.headers.Authorization = `Bearer ${token}`;
  return cfg;
});

http.interceptors.response.use(
  res => res,
  err => {
    // 统一错误提示/上报
    console.error('API Error:', err?.response?.status, err?.response?.data);
    return Promise.reject(err);
  }
);