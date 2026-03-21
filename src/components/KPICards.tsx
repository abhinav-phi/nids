import { useState, useEffect, useRef } from "react";
import { getStats } from "@/api/client";
import { Activity, AlertTriangle, Clock, Shield } from "lucide-react";

interface Stats {
  total_flows: number;
  total_attacks: number;
  uptime_seconds: number;
  benign_count: number;
}

const formatUptime = (s: number) => {
  const h = Math.floor(s / 3600).toString().padStart(2, "0");
  const m = Math.floor((s % 3600) / 60).toString().padStart(2, "0");
  const sec = Math.floor(s % 60).toString().padStart(2, "0");
  return `${h}:${m}:${sec}`;
};

const KPICards = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [flash, setFlash] = useState(false);
  const prevRef = useRef<Stats | null>(null);

  useEffect(() => {
    const fetch = () => {
      getStats()
        .then((data) => {
          if (
            prevRef.current &&
            JSON.stringify(data) !== JSON.stringify(prevRef.current)
          ) {
            setFlash(true);
            setTimeout(() => setFlash(false), 400);
          }
          prevRef.current = data;
          setStats(data);
          setLoading(false);
        })
        .catch(() => setLoading(false));
    };
    fetch();
    const id = setInterval(fetch, 10000);
    return () => clearInterval(id);
  }, []);

  const cards = stats
    ? [
        { label: "Flows Analyzed", value: stats.total_flows.toLocaleString(), icon: Activity, color: "text-primary" },
        { label: "Attacks Detected", value: stats.total_attacks.toLocaleString(), icon: AlertTriangle, color: "text-severity-critical" },
        { label: "System Uptime", value: formatUptime(stats.uptime_seconds), icon: Clock, color: "text-severity-high" },
        { label: "Benign Traffic", value: stats.benign_count.toLocaleString(), icon: Shield, color: "text-severity-medium" },
      ]
    : [];

  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-card border border-border rounded-lg p-4 animate-pulse h-24" />
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div
          key={c.label}
          className="bg-card border border-border rounded-lg p-4 flex items-center gap-3"
        >
          <c.icon className={`w-8 h-8 ${c.color} shrink-0`} />
          <div>
            <div className={`text-2xl font-bold text-foreground ${flash ? "animate-pulse-number" : ""}`}>
              {c.value}
            </div>
            <div className="text-xs text-muted-foreground">{c.label}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default KPICards;
