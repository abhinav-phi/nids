import { useState, useEffect, useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { getAlerts } from "@/api/client";
const SEVERITY_COLORS: Record<string, string> = {
  CRITICAL: "#ff716c",
  HIGH:     "#699cff",
  MEDIUM:   "#ac8aff",
  LOW:      "rgba(255,255,255,0.2)",
};
const SEVERITY_RANK: Record<string, number> = {
  CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1,
};
const AttackTimeline = () => {
  const [alerts, setAlerts]   = useState<any[]>([]);
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
    <div
      className="rounded-2xl p-8"
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
            24-Hour Threat Trajectory
          </h2>
          <p className="text-sm mt-1" style={{ color: "rgba(255,255,255,0.4)" }}>
            Detected attack attempts categorized by hourly clusters
          </p>
        </div>
        {}
        <div className="flex items-center gap-4">
          {[
            { color: "#a1faff", label: "Benign Flow" },
            { color: "#ff716c", label: "Intrusion" },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm" style={{ background: color }} />
              <span className="text-[10px] font-bold uppercase tracking-widest"
                style={{ color: "rgba(255,255,255,0.4)" }}>
                {label}
              </span>
            </div>
          ))}
        </div>
      </div>
      {}
      {loading ? (
        <div className="h-52 rounded-xl animate-pulse" style={{ background: "rgba(255,255,255,0.04)" }} />
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={data} barCategoryGap="20%">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
            <XAxis
              dataKey="hour"
              tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              axisLine={false} tickLine={false}
            />
            <YAxis
              tick={{ fill: "rgba(255,255,255,0.3)", fontSize: 10 }}
              axisLine={false} tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background:   "rgba(10,14,25,0.95)",
                border:       "1px solid rgba(161,250,255,0.2)",
                borderRadius: "8px",
                color:        "#e8eafb",
                fontSize:     12,
              }}
              cursor={{ fill: "rgba(255,255,255,0.03)" }}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} name="Attacks">
              {data.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.count === 0
                    ? "rgba(161,250,255,0.12)"
                    : SEVERITY_COLORS[entry.maxSeverity] || "rgba(255,255,255,0.2)"}
                  fillOpacity={entry.count === 0 ? 0.5 : 0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};
export default AttackTimeline;
