import { useState, useEffect } from "react";
import { getIPLeaderboard } from "@/api/client";

interface IPEntry {
  ip: string;
  attack_count: number;
  last_seen: string;
  top_attack_type?: string;
}

const RANK_COLORS = ["#e3b341", "#8b949e", "#cd7f32"];

const relativeTime = (ts: string) => {
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min${mins > 1 ? "s" : ""} ago`;
  const hrs = Math.floor(mins / 60);
  return `${hrs} hr${hrs > 1 ? "s" : ""} ago`;
};

const IPLeaderboard = () => {
  const [data, setData] = useState<IPEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = () => {
      getIPLeaderboard()
        .then((d) => {
          setData(Array.isArray(d) ? d : d.leaderboard || []);
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetch();
    const id = setInterval(fetch, 30000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="bg-card border border-border rounded-lg p-4 h-full">
      <h3 className="text-sm font-semibold text-foreground mb-3">Top Attacker IPs</h3>
      {loading ? (
        <div className="h-48 animate-pulse bg-secondary rounded" />
      ) : data.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-muted-foreground text-sm">
          No data yet
        </div>
      ) : (
        <table className="w-full text-xs">
          <thead>
            <tr className="text-muted-foreground border-b border-border">
              <th className="text-left py-2 px-2">Rank</th>
              <th className="text-left py-2 px-2">IP Address</th>
              <th className="text-left py-2 px-2">Attacks</th>
              <th className="text-left py-2 px-2">Last Seen</th>
              <th className="text-left py-2 px-2">Badge</th>
            </tr>
          </thead>
          <tbody>
            {data.slice(0, 10).map((entry, i) => (
              <tr key={entry.ip} className="border-b border-border/50 hover:bg-secondary/30">
                <td className="py-1.5 px-2">
                  <span
                    className="inline-flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold"
                    style={{
                      backgroundColor: i < 3 ? RANK_COLORS[i] + "33" : "transparent",
                      color: i < 3 ? RANK_COLORS[i] : "#8b949e",
                    }}
                  >
                    {i + 1}
                  </span>
                </td>
                <td className="py-1.5 px-2 font-mono-code text-foreground">{entry.ip}</td>
                <td className="py-1.5 px-2 text-foreground font-semibold">{entry.attack_count}</td>
                <td className="py-1.5 px-2 text-muted-foreground">{relativeTime(entry.last_seen)}</td>
                <td className="py-1.5 px-2">
                  <span className="px-2 py-0.5 rounded bg-secondary text-muted-foreground text-[10px]">
                    {entry.top_attack_type || "—"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default IPLeaderboard;
