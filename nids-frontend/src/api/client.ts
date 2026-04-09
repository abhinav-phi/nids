import axios from "axios";
const api = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 10000,
});
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (!err.response) {
      console.debug("[API] Backend offline:", err.message);
    }
    return Promise.reject(err);
  }
);
export const getAlerts = (params?: Record<string, unknown>) =>
  api.get("/api/alerts", { params }).then((r) => r.data);
export const getStats = () =>
  api.get("/api/stats").then((r) => r.data);
export const getIPLeaderboard = () =>
  api.get("/api/ip-leaderboard").then((r) => r.data);
export const checkHealth = () =>
  api.get("/health").then((r) => r.data);
export default api;
