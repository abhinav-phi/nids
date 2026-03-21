import { useState, useEffect } from "react";
import { checkHealth } from "@/api/client";

interface StatusBarProps {
  wsConnected: boolean;
}

const StatusBar = ({ wsConnected }: StatusBarProps) => {
  const [online, setOnline] = useState(false);
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const check = () => {
      checkHealth()
        .then(() => setOnline(true))
        .catch(() => setOnline(false));
    };
    check();
    const id = setInterval(check, 10000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="h-14 bg-card border-b border-border flex items-center justify-between px-4 sticky top-0 z-50">
      <div className="flex items-center gap-3">
        <span className="text-primary font-bold text-xl tracking-tight">NIDS</span>
        <span className="text-muted-foreground text-sm hidden sm:inline">
          Network Intrusion Detection System
        </span>
      </div>
      <div className="flex items-center gap-4">
        {!wsConnected && (
          <span className="text-severity-high text-xs animate-pulse">
            ⚡ Reconnecting...
          </span>
        )}
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              online ? "bg-severity-medium" : "bg-severity-critical"
            }`}
          />
          <span
            className={`text-xs font-medium ${
              online ? "text-severity-medium" : "text-severity-critical"
            }`}
          >
            {online ? "SYSTEM ACTIVE" : "OFFLINE"}
          </span>
        </div>
        <span className="text-muted-foreground text-xs font-mono-code">
          {time.toLocaleTimeString()}
        </span>
      </div>
    </header>
  );
};

export default StatusBar;
