import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { Alert } from "@/hooks/useWebSocket";

interface Props {
  alertHistory: Alert[];
}

const TrafficChart = ({ alertHistory }: Props) => {
  const data = useMemo(() => {
    if (!alertHistory.length) return [];
    const buckets: Record<string, number> = {};
    alertHistory.forEach((a) => {
      const d = new Date(a.timestamp);
      const key = d.toLocaleTimeString("en-US", { hour12: false });
      buckets[key] = (buckets[key] || 0) + 1;
    });
    return Object.entries(buckets)
      .map(([time, count]) => ({ time, count }))
      .slice(-60);
  }, [alertHistory]);

  return (
    <div className="bg-card border border-border rounded-lg p-4 h-full">
      <h3 className="text-sm font-semibold text-foreground mb-3">Live Traffic Activity</h3>
      {data.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          No data yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="blueGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#58a6ff" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#58a6ff" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis dataKey="time" tick={{ fill: "#8b949e", fontSize: 10 }} />
            <YAxis tick={{ fill: "#8b949e", fontSize: 10 }} />
            <Tooltip
              contentStyle={{ backgroundColor: "#161b22", border: "1px solid #30363d", color: "#fff" }}
            />
            <Area type="monotone" dataKey="count" stroke="#58a6ff" fill="url(#blueGrad)" />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default TrafficChart;
