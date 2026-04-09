import { useState, useEffect } from "react";
import { checkHealth } from "@/api/client";
interface StatusBarProps {
  wsConnected: boolean;
}
const StatusBar = ({ wsConnected }: StatusBarProps) => {
  const [online, setOnline] = useState(false);
  const [time, setTime]     = useState(new Date());
  useEffect(() => {
    const check = () => {
      checkHealth().then(() => setOnline(true)).catch(() => setOnline(false));
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
    <header
      className="sticky top-0 z-50 flex items-center justify-between px-8 py-4"
      style={{
        background: "rgba(10,14,25,0.92)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        boxShadow: "0 0 20px rgba(0,245,255,0.04)",
      }}
    >
      {}
      <div className="flex items-center gap-6">
        {}
        <span
          className="lg:hidden font-bold text-xl"
          style={{
            color: "#a1faff",
            fontFamily: "'Space Grotesk', sans-serif",
            letterSpacing: "-0.02em",
          }}
        >
          The Sentinel
        </span>
        {}
        <span
          className="hidden lg:block font-bold text-lg"
          style={{
            color: "rgba(255,255,255,0.5)",
            fontFamily: "'Space Grotesk', sans-serif",
          }}
        >
          Dashboard
        </span>
        {}
        <div
          className="flex items-center gap-2 px-3 py-1 rounded-full"
          style={{
            background: online ? "rgba(161,250,255,0.08)" : "rgba(248,81,73,0.08)",
            border: `1px solid ${online ? "rgba(161,250,255,0.2)" : "rgba(248,81,73,0.2)"}`,
          }}
        >
          <span
            className={`w-2 h-2 rounded-full ${online ? "animate-blink-dot" : ""}`}
            style={{ backgroundColor: online ? "#a1faff" : "#f85149" }}
          />
          <span
            className="text-[10px] font-bold uppercase tracking-widest"
            style={{ color: online ? "#a1faff" : "#f85149" }}
          >
            {online ? "System Active" : "Offline"}
          </span>
        </div>
        {}
        {!wsConnected && (
          <span
            className="text-xs flex items-center gap-1.5 animate-pulse"
            style={{ color: "#e3b341" }}
          >
            ⚡ Reconnecting...
          </span>
        )}
      </div>
      {}
      <div className="flex items-center gap-6">
        <div className="hidden xl:flex flex-col items-end">
          <span className="text-[10px] uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.3)" }}>
            Local Time
          </span>
          <span
            className="text-sm font-bold font-mono-code"
            style={{ color: "#a1faff" }}
          >
            {time.toLocaleTimeString("en-US", { hour12: false })} UTC
          </span>
        </div>
        {}
        <span
          className="xl:hidden font-mono-code text-xs"
          style={{ color: "rgba(255,255,255,0.35)" }}
        >
          {time.toLocaleTimeString()}
        </span>
      </div>
    </header>
  );
};
export default StatusBar;
