import { useState, useEffect } from "react";
import { getIPLeaderboard } from "@/api/client";
import { Skull } from "lucide-react";
interface IPEntry {
  ip?:             string;
  source_ip?:      string;
  attack_count:    number;
  last_seen:       string;
  top_attack_type?: string;
}
const relativeTime = (ts: string) => {
  if (!ts) return "—";
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1)  return "just now";
  if (mins < 60) return `${mins}m ago`;
  return `${Math.floor(mins / 60)}h ago`;
};
const RANK_STYLE = [
  { border: "#ff716c", bg: "rgba(255,113,108,0.05)", badge: "Critical Vector", badgeColor: "#ff716c", badgeBg: "rgba(255,113,108,0.12)" },
  { border: "#699cff", bg: "rgba(105,156,255,0.04)", badge: "Port Scanner",    badgeColor: "#699cff", badgeBg: "rgba(105,156,255,0.12)" },
  { border: "rgba(255,255,255,0.12)", bg: "rgba(255,255,255,0.03)", badge: "Infiltration", badgeColor: "rgba(255,255,255,0.4)", badgeBg: "rgba(255,255,255,0.06)" },
];
const IPLeaderboard = () => {
  const [data, setData]       = useState<IPEntry[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const fetchData = () => {
      getIPLeaderboard()
        .then((d) => {
          setData(Array.isArray(d) ? d : d.leaderboard || []);
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetchData();
    const id = setInterval(fetchData, 30000);
    return () => clearInterval(id);
  }, []);
  return (
    <div
      className="rounded-2xl p-6 flex flex-col h-full"
      style={{
        background:   "rgba(26,31,46,0.6)",
        backdropFilter: "blur(12px)",
        border:       "1px solid rgba(255,255,255,0.06)",
        boxShadow:    "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      {}
      <div className="flex items-center gap-2 mb-6">
        <Skull size={18} style={{ color: "#ff716c" }} />
        <span
          className="text-lg font-bold"
          style={{ color: "#e8eafb", fontFamily: "'Space Grotesk', sans-serif" }}
        >
          Top Adversaries
        </span>
        {data.length > 0 && (
          <span
            className="ml-auto text-[10px] font-bold px-2 py-0.5 rounded-full"
            style={{
              background: "rgba(255,113,108,0.1)",
              color:      "#ff716c",
              border:     "1px solid rgba(255,113,108,0.2)",
            }}
          >
            {data.length} tracked
          </span>
        )}
      </div>
      {}
      {loading ? (
        <div className="space-y-3 flex-1">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-16 rounded-lg animate-pulse"
              style={{ background: "rgba(255,255,255,0.04)" }} />
          ))}
        </div>
      ) : data.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2"
          style={{ color: "rgba(255,255,255,0.2)" }}>
          <span className="text-2xl">🔍</span>
          <span className="text-sm">No attackers tracked yet</span>
        </div>
      ) : (
        <div className="space-y-3 overflow-y-auto flex-1">
          {data.slice(0, 5).map((entry, i) => {
            const ip    = entry.ip || entry.source_ip || "unknown";
            const rs    = RANK_STYLE[Math.min(i, RANK_STYLE.length - 1)];
            return (
              <div
                key={ip + i}
                className="flex items-center justify-between p-3 rounded-lg transition-all hover:scale-[1.01]"
                style={{
                  background:  rs.bg,
                  borderLeft:  `2px solid ${rs.border}`,
                  border:      `1px solid ${rs.border}22`,
                  borderLeftWidth: "2px",
                  borderLeftColor: rs.border,
                }}
              >
                <div className="flex flex-col min-w-0">
                  <span
                    className="text-sm font-mono-code font-bold truncate"
                    style={{ color: i === 0 ? "#ff716c" : "#e8eafb" }}
                  >
                    {ip}
                  </span>
                  <span
                    className="text-[10px] font-bold uppercase mt-0.5"
                    style={{ color: rs.badgeColor }}
                  >
                    {rs.badge}
                  </span>
                </div>
                <span
                  className="px-2 py-1 rounded text-xs font-bold shrink-0 ml-3"
                  style={{
                    background: rs.badgeBg,
                    color:      rs.badgeColor,
                  }}
                >
                  {entry.attack_count} Hits
                </span>
              </div>
            );
          })}
        </div>
      )}
      {}
      <div className="mt-4 pt-4" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
        <button
          className="w-full py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-colors"
          style={{
            background: "rgba(255,255,255,0.04)",
            border:     "1px solid rgba(255,255,255,0.08)",
            color:      "rgba(255,255,255,0.4)",
          }}
          onMouseEnter={e => {
            (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.08)";
            (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.7)";
          }}
          onMouseLeave={e => {
            (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)";
            (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.4)";
          }}
        >
          Open Global Blocklist
        </button>
      </div>
    </div>
  );
};
export default IPLeaderboard;
