import { useMemo, useState, useEffect, useRef } from "react";
import {
  AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import type { Alert } from "@/hooks/useWebSocket";
import { getStats } from "@/api/client";
interface Props {
  alertHistory: Alert[];
}
interface DataPoint {
  time:   string;
  alerts: number;
  flows:  number;
}
const TrafficChart = ({ alertHistory }: Props) => {
  const [flowHistory, setFlowHistory] = useState<{ time: string; flows: number }[]>([]);
  const prevFlows = useRef(0);
  useEffect(() => {
    const tick = () => {
      getStats()
        .then((data) => {
          const newFlows = (data.total_flows || 0) - prevFlows.current;
          prevFlows.current = data.total_flows || 0;
          const time = new Date().toLocaleTimeString("en-US", { hour12: false });
          setFlowHistory((prev) => [...prev, { time, flows: Math.max(0, newFlows) }].slice(-60));
        })
        .catch(() => {});
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => clearInterval(id);
  }, []);
  const alertBuckets = useMemo(() => {
    const buckets: Record<string, number> = {};
    alertHistory.forEach((a) => {
      const key = new Date(a.timestamp).toLocaleTimeString("en-US", { hour12: false });
      buckets[key] = (buckets[key] || 0) + 1;
    });
    return buckets;
  }, [alertHistory]);
  const data: DataPoint[] = useMemo(() => {
    return flowHistory.map((f) => ({
      time:   f.time,
      flows:  f.flows,
      alerts: alertBuckets[f.time] || 0,
    }));
  }, [flowHistory, alertBuckets]);
  return (
    <div
      className="rounded-2xl p-8 relative overflow-hidden flex flex-col"
      style={{
        background:   "rgba(26,31,46,0.6)",
        backdropFilter: "blur(12px)",
        border:       "1px solid rgba(255,255,255,0.06)",
        boxShadow:    "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      {}
      <div className="flex items-start justify-between mb-8">
        <div>
          <h2
            className="text-xl font-bold"
            style={{ color: "#e8eafb", fontFamily: "'Space Grotesk', sans-serif" }}
          >
            Real-Time Traffic Volatility
          </h2>
          <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
            Live packets per second across all monitored interfaces
          </p>
        </div>
        <div className="flex gap-2">
          <span
            className="px-3 py-1 rounded-md text-[10px] font-bold uppercase"
            style={{
              background: "rgba(255,255,255,0.05)",
              border:     "1px solid rgba(255,255,255,0.1)",
              color:      "rgba(255,255,255,0.5)",
            }}
          >
            Live Feed
          </span>
        </div>
      </div>
      {}
      {data.length === 0 ? (
        <div
          className="flex items-center justify-center h-52 rounded-xl"
          style={{ background: "rgba(255,255,255,0.02)", border: "1px dashed rgba(255,255,255,0.06)" }}
        >
          <div className="flex flex-col items-center gap-2" style={{ color: "rgba(255,255,255,0.2)" }}>
            <span className="text-2xl">📡</span>
            <span className="text-sm">No traffic data yet — waiting for flows...</span>
          </div>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="cyanGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#a1faff" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#a1faff" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="redGrad2" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%"   stopColor="#ff716c" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#ff716c" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis
              dataKey="time"
              tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              axisLine={false} tickLine={false}
            />
            <YAxis
              tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              axisLine={false} tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "rgba(10,14,25,0.95)",
                border:     "1px solid rgba(161,250,255,0.2)",
                borderRadius: "8px",
                color:      "#e8eafb",
                fontSize:   12,
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: 11, color: "rgba(255,255,255,0.4)" }}
            />
            <Area
              type="monotone" dataKey="flows"
              name="Safe Traffic"
              stroke="#a1faff" strokeWidth={2}
              fill="url(#cyanGrad)"
            />
            <Area
              type="monotone" dataKey="alerts"
              name="Anomaly Detected"
              stroke="#ff716c" strokeWidth={2}
              fill="url(#redGrad2)"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};
export default TrafficChart;
