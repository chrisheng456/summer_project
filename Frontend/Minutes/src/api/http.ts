import axios from "axios";
export const http = axios.create({
  baseURL: "/api",
  timeout: 0,
});

// Request interceptor: automatically attach token
http.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Response interceptor: unified handling of 401/403
http.interceptors.response.use(
  (res) => res,
  (err) => {
    const code = err?.response?.status;
    if (code === 401 || code === 403) {
      localStorage.removeItem("token");
    }
    return Promise.reject(err);
  }
)