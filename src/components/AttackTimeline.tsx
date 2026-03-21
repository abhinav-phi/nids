import { useState, useEffect, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { getAlerts } from "@/api/client";

const SEVERITY_COLORS: Record<string, string> = {
  CRITICAL: "#f85149",
  HIGH: "#e3b341",
  MEDIUM: "#3fb950",
  LOW: "#8b949e",
};

const SEVERITY_RANK: Record<string, number> = {
  CRITICAL: 4,
  HIGH: 3,
  MEDIUM: 2,
  LOW: 1,
};

const AttackTimeline = () => {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = () => {
      getAlerts()
        .then((data) => {
          setAlerts(Array.isArray(data) ? data : data.alerts || []);
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetch();
    const id = setInterval(fetch, 30000);
    return () => clearInterval(id);
  }, []);

  const data = useMemo(() => {
    const now = new Date();
    const hours: { hour: string; count: number; maxSeverity: string }[] = [];
    for (let i = 11; i >= 0; i--) {
      const h = new Date(now.getTime() - i * 3600000);
      const label = h.toLocaleTimeString("en-US", { hour: "2-digit", hour12: false });
      const hourAlerts = alerts.filter((a) => {
        const d = new Date(a.timestamp);
        return d.getHours() === h.getHours() && now.getTime() - d.getTime() < 12 * 3600000;
      });
      let maxSev = "LOW";
      hourAlerts.forEach((a) => {
        const sev = (a.severity || "LOW").toUpperCase();
        if ((SEVERITY_RANK[sev] || 0) > (SEVERITY_RANK[maxSev] || 0)) maxSev = sev;
      });
      hours.push({ hour: label, count: hourAlerts.length, maxSeverity: maxSev });
    }
    return hours;
  }, [alerts]);

  return (
    <div className="bg-card border border-border rounded-lg p-4 h-full">
      <h3 className="text-sm font-semibold text-foreground mb-3">Attack Timeline (Last 12 Hours)</h3>
      {loading ? (
        <div className="h-48 animate-pulse bg-secondary rounded" />
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
            <XAxis dataKey="hour" tick={{ fill: "#8b949e", fontSize: 10 }} />
            <YAxis tick={{ fill: "#8b949e", fontSize: 10 }} />
            <Tooltip contentStyle={{ backgroundColor: "#161b22", border: "1px solid #30363d", color: "#fff" }} />
            <Bar dataKey="count" radius={[4, 4, 0, 0]}>
              {data.map((entry, i) => (
                <Cell key={i} fill={SEVERITY_COLORS[entry.maxSeverity] || "#8b949e"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default AttackTimeline;
