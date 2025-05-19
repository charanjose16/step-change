import axios from "axios";
import { jwtDecode } from "jwt-decode";

const axiosInstance = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL || "http://localhost:8000/api/v1",
  headers: {
    "Content-Type": "application/json",
  },
});

// Request interceptor: check token expiry before each request.
axiosInstance.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) {
    try {
      const { exp } = jwtDecode(token);
      // If token expires in less than 2 minutes, clear it.
      if (exp - Date.now() / 1000 < 120) {
        localStorage.removeItem("access_token");
      } else {
        config.headers.Authorization = `Bearer ${token}`;
      }
    } catch (error) {
      console.error("Error decoding token:", error);
      localStorage.removeItem("access_token");
    }
  }
  return config;
});

// Response interceptor: if a 401 error occurs, clear token & redirect to login.
axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem("access_token");
    }
    return Promise.reject(error);
  }
);

export default axiosInstance;