import { useState, useEffect } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Legend,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { getStats } from "@/api/client";

const COLORS = ["#58a6ff", "#f85149", "#e3b341", "#3fb950", "#8b949e", "#bc8cff", "#f778ba"];

const AttackPieChart = () => {
  const [data, setData] = useState<{ name: string; value: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = () => {
      getStats()
        .then((s) => {
          if (s.attacks_by_type) {
            const entries = Object.entries(s.attacks_by_type).map(
              ([name, value]) => ({ name, value: value as number })
            );
            setData(entries);
          }
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetch();
    const id = setInterval(fetch, 15000);
    return () => clearInterval(id);
  }, []);

  const total = data.reduce((s, d) => s + d.value, 0);

  return (
    <div className="bg-card border border-border rounded-lg p-4 h-full">
      <h3 className="text-sm font-semibold text-foreground mb-3">Attack Type Distribution</h3>
      {loading ? (
        <div className="h-48 animate-pulse bg-secondary rounded" />
      ) : data.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          No data yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={data}
              cx="40%"
              cy="50%"
              innerRadius={50}
              outerRadius={80}
              dataKey="value"
              label={false}
            >
              {data.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Pie>
            <text x="40%" y="50%" textAnchor="middle" dominantBaseline="middle" fill="#fff" fontSize={18} fontWeight="bold">
              {total}
            </text>
            <Tooltip contentStyle={{ backgroundColor: "#161b22", border: "1px solid #30363d", color: "#fff" }} />
            <Legend
              layout="vertical"
              align="right"
              verticalAlign="middle"
              wrapperStyle={{ fontSize: 11, color: "#8b949e" }}
            />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default AttackPieChart;
