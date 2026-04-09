import { useState, useEffect, useRef } from "react";
import { getStats } from "@/api/client";
import { Activity, AlertTriangle, Clock, Shield } from "lucide-react";
interface Stats {
  total_flows:    number;
  total_attacks:  number;
  uptime_seconds: number;
  benign_count:   number;
}
const formatUptime = (s: number) => {
  const h   = Math.floor(s / 3600).toString().padStart(2, "0");
  const m   = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${h}:${m}:${sec}`;
};
const CARDS = [
  {
    label:     "Total Network Flows",
    sublabel:  "Live Monitor",
    icon:      Activity,
    accent:    "#a1faff",      
    bg:        "rgba(161,250,255,0.06)",
    border:    "#a1faff",
    glow:      "rgba(161,250,255,0.08)",
    iconBg:    "rgba(161,250,255,0.1)",
  },
  {
    label:     "Attacks Detected",
    sublabel:  "Detection Engine",
    icon:      AlertTriangle,
    accent:    "#ff716c",      
    bg:        "rgba(255,113,108,0.06)",
    border:    "#ff716c",
    glow:      "rgba(255,113,108,0.1)",
    iconBg:    "rgba(255,113,108,0.1)",
  },
  {
    label:     "System Uptime",
    sublabel:  "Service Health",
    icon:      Clock,
    accent:    "#699cff",      
    bg:        "rgba(105,156,255,0.06)",
    border:    "#699cff",
    glow:      "rgba(105,156,255,0.08)",
    iconBg:    "rgba(105,156,255,0.1)",
  },
  {
    label:     "Benign Traffic",
    sublabel:  "ML Confidence",
    icon:      Shield,
    accent:    "#ac8aff",      
    bg:        "rgba(172,138,255,0.06)",
    border:    "#ac8aff",
    glow:      "rgba(172,138,255,0.08)",
    iconBg:    "rgba(172,138,255,0.1)",
  },
];
const KPICards = () => {
  const [stats, setStats]     = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [flash, setFlash]     = useState(false);
  const prevRef               = useRef<Stats | null>(null);
  useEffect(() => {
    const fetchData = () => {
      getStats()
        .then((data) => {
          if (prevRef.current && JSON.stringify(data) !== JSON.stringify(prevRef.current)) {
            setFlash(true);
            setTimeout(() => setFlash(false), 400);
          }
          prevRef.current = data;
          setStats(data);
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetchData();
    const id = setInterval(fetchData, 10000);
    return () => clearInterval(id);
  }, []);
  const values = stats
    ? [
        stats.total_flows.toLocaleString(),
        stats.total_attacks.toLocaleString(),
        formatUptime(stats.uptime_seconds),
        stats.benign_count.toLocaleString(),
      ]
    : ["—", "—", "—:—:—", "—"];
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
        {[...Array(4)].map((_, i) => (
          <div
            key={i}
            className="h-28 rounded-xl animate-pulse"
            style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}
          />
        ))}
      </div>
    );
  }
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
      {CARDS.map((c, i) => {
        const Icon = c.icon;
        return (
          <div
            key={c.label}
            className="rounded-xl p-6 flex flex-col gap-4 transition-all duration-200 hover:scale-[1.02] hover:brightness-110"
            style={{
              background: c.bg,
              borderLeft: `4px solid ${c.border}`,
              border: `1px solid rgba(255,255,255,0.06)`,
              borderLeftWidth: "4px",
              borderLeftColor: c.border,
              boxShadow: `0 0 24px ${c.glow}, 0 4px 16px rgba(0,0,0,0.2)`,
              backdropFilter: "blur(12px)",
            }}
          >
            {}
            <div className="flex items-start justify-between">
              <div
                className="w-11 h-11 rounded-lg flex items-center justify-center"
                style={{ background: c.iconBg }}
              >
                <Icon size={20} style={{ color: c.accent }} />
              </div>
              <span
                className="text-[10px] font-bold uppercase tracking-widest"
                style={{ color: "rgba(255,255,255,0.3)" }}
              >
                {c.sublabel}
              </span>
            </div>
            {}
            <div>
              <p className="text-xs font-medium mb-1" style={{ color: "rgba(255,255,255,0.5)" }}>
                {c.label}
              </p>
              <p
                className={`text-3xl font-bold leading-none ${flash ? "animate-pulse-number" : ""}`}
                style={{
                  color: c.accent,
                  fontFamily: "'Space Grotesk', sans-serif",
                }}
              >
                {values[i]}
              </p>
            </div>
          </div>
        );
      })}
    </div>
  );
};
export default KPICards;
