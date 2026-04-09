import { useState, useEffect } from "react";
import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from "recharts";
import { getStats } from "@/api/client";
const COLORS = ["#a1faff", "#699cff", "#ac8aff", "#ff716c", "#3fb950", "#e3b341", "rgba(255,255,255,0.4)"];
const AttackPieChart = () => {
  const [data, setData]       = useState<{ name: string; value: number }[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const fetchData = () => {
      getStats()
        .then((s) => {
          if (s.attacks_by_type) {
            setData(
              Object.entries(s.attacks_by_type).map(([name, value]) => ({
                name,
                value: value as number,
              }))
            );
          }
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetchData();
    const id = setInterval(fetchData, 15000);
    return () => clearInterval(id);
  }, []);
  const total = data.reduce((s, d) => s + d.value, 0);
  return (
    <div
      className="rounded-2xl p-8 flex flex-col h-full"
      style={{
        background:   "rgba(26,31,46,0.6)",
        backdropFilter: "blur(12px)",
        border:       "1px solid rgba(255,255,255,0.06)",
        boxShadow:    "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      {}
      <div className="flex items-center justify-between mb-6">
        <h2
          className="text-xl font-bold"
          style={{ color: "#e8eafb", fontFamily: "'Space Grotesk', sans-serif" }}
        >
          Threat Distribution
        </h2>
        {total > 0 && (
          <span
            className="text-[10px] font-bold px-2 py-0.5 rounded-full"
            style={{
              background: "rgba(161,250,255,0.08)",
              color:      "#a1faff",
              border:     "1px solid rgba(161,250,255,0.2)",
            }}
          >
            {total} total
          </span>
        )}
      </div>
      {loading ? (
        <div className="flex-1 h-48 rounded-xl animate-pulse"
          style={{ background: "rgba(255,255,255,0.04)" }} />
      ) : data.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2"
          style={{ color: "rgba(255,255,255,0.2)" }}>
          <span className="text-3xl">📊</span>
          <span className="text-sm">No attack data yet</span>
        </div>
      ) : (
        <>
          {}
          <div className="relative flex justify-center py-4">
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={data}
                  cx="40%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  label={false}
                >
                  {data.map((_, i) => (
                    <Cell
                      key={i}
                      fill={COLORS[i % COLORS.length]}
                      stroke="rgba(0,0,0,0.4)"
                      strokeWidth={1}
                    />
                  ))}
                </Pie>
                {}
                <text x="40%" y="48%" textAnchor="middle" dominantBaseline="middle"
                  fill="#e8eafb" fontSize={26} fontWeight="700"
                  style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                  {total}
                </text>
                <text x="40%" y="60%" textAnchor="middle" dominantBaseline="middle"
                  fill="rgba(255,255,255,0.3)" fontSize={11}>
                  Active Alerts
                </text>
                <Tooltip
                  contentStyle={{
                    background:   "rgba(10,14,25,0.95)",
                    border:       "1px solid rgba(161,250,255,0.2)",
                    borderRadius: "8px",
                    color:        "#e8eafb",
                    fontSize:     12,
                  }}
                />
                <Legend
                  layout="vertical"
                  align="right"
                  verticalAlign="middle"
                  formatter={(value) => (
                    <span style={{ color: "rgba(255,255,255,0.55)", fontSize: 11 }}>{value}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          {}
          <div className="mt-2 space-y-2">
            {data.slice(0, 4).map((d, i) => (
              <div key={d.name} className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ background: COLORS[i % COLORS.length] }}
                  />
                  <span style={{ color: "rgba(255,255,255,0.6)" }}>{d.name}</span>
                </div>
                <span className="font-mono-code" style={{ color: "#e8eafb" }}>
                  {Math.round(d.value / total * 100)}%
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
};
export default AttackPieChart;
