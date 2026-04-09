import { useState, useEffect, useRef } from "react";
import type { Alert } from "@/hooks/useWebSocket";
import SHAPExplainer from "./SHAPExplainer";
import { ChevronRight } from "lucide-react";
interface Props {
  alertHistory: Alert[];
}
const SEV_STYLES: Record<string, { color: string; bg: string; border: string; label: string }> = {
  CRITICAL: {
    color:  "#ff716c",
    bg:     "rgba(255,113,108,0.12)",
    border: "rgba(255,113,108,0.25)",
    label:  "Critical",
  },
  HIGH: {
    color:  "#699cff",
    bg:     "rgba(0,90,194,0.2)",
    border: "rgba(105,156,255,0.25)",
    label:  "High",
  },
  MEDIUM: {
    color:  "#ac8aff",
    bg:     "rgba(143,96,250,0.12)",
    border: "rgba(172,138,255,0.2)",
    label:  "Medium",
  },
  LOW: {
    color:  "rgba(255,255,255,0.4)",
    bg:     "rgba(255,255,255,0.05)",
    border: "rgba(255,255,255,0.1)",
    label:  "Low",
  },
  NONE: {
    color:  "rgba(255,255,255,0.25)",
    bg:     "rgba(255,255,255,0.03)",
    border: "rgba(255,255,255,0.06)",
    label:  "None",
  },
};
const AlertFeed = ({ alertHistory }: Props) => {
  const [selected, setSelected] = useState<Alert | null>(null);
  const [newIds, setNewIds]     = useState<Set<string>>(new Set());
  const prevLenRef              = useRef(0);
  const alerts                  = alertHistory.slice(0, 50);
  useEffect(() => {
    if (alertHistory.length > prevLenRef.current) {
      const fresh = alertHistory.slice(0, alertHistory.length - prevLenRef.current);
      const ids   = new Set(fresh.map((a) => a.timestamp + a.src_ip));
      setNewIds(ids);
      const t = setTimeout(() => setNewIds(new Set()), 600);
      prevLenRef.current = alertHistory.length;
      return () => clearTimeout(t);
    }
  }, [alertHistory]);
  return (
    <div
      className="rounded-2xl border flex flex-col h-full overflow-hidden"
      style={{
        background:  "rgba(26,31,46,0.6)",
        backdropFilter: "blur(12px)",
        borderColor: "rgba(255,255,255,0.06)",
        boxShadow:   "0 4px 24px rgba(0,0,0,0.3)",
      }}
    >
      {}
      <div
        className="flex items-center justify-between px-8 py-5 shrink-0"
        style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.03)" }}
      >
        <div className="flex items-center gap-3">
          <span
            className="w-2 h-2 rounded-full animate-blink-dot"
            style={{ backgroundColor: alerts.length > 0 ? "#ff716c" : "#a1faff" }}
          />
          <span
            className="text-lg font-bold"
            style={{ color: "#e8eafb", fontFamily: "'Space Grotesk', sans-serif" }}
          >
            Live Threat Log
          </span>
          {alerts.length > 0 && (
            <span
              className="text-[10px] font-bold px-2 py-0.5 rounded-full"
              style={{
                background: "rgba(255,113,108,0.12)",
                color:      "#ff716c",
                border:     "1px solid rgba(255,113,108,0.2)",
              }}
            >
              {alerts.length} alerts
            </span>
          )}
        </div>
        <button
          className="text-xs font-bold transition-colors hover:underline"
          style={{ color: "#a1faff" }}
        >
          View Historical Archive
        </button>
      </div>
      {}
      {alerts.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-3"
          style={{ color: "rgba(255,255,255,0.2)" }}>
          <span className="text-3xl">🛡️</span>
          <span className="text-sm">No threats detected</span>
        </div>
      ) : (
        <div className="overflow-auto flex-1">
          <table className="w-full text-left border-collapse">
            <thead
              className="sticky top-0 z-10"
              style={{ background: "rgba(26,31,46,0.98)" }}
            >
              <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                {["Timestamp", "Source IP", "Attack Type", "Severity", "Action"].map((h, idx) => (
                  <th
                    key={h}
                    className="py-4 text-[10px] font-bold uppercase tracking-widest"
                    style={{
                      color:   "rgba(255,255,255,0.3)",
                      paddingLeft:  idx === 0 ? "2rem" : "1rem",
                      paddingRight: idx === 4 ? "2rem" : "1rem",
                      textAlign: idx === 4 ? "right" : "left",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert, i) => {
                const sev       = (alert.severity || "LOW").toUpperCase();
                const style     = SEV_STYLES[sev] || SEV_STYLES.LOW;
                const id        = alert.timestamp + alert.src_ip;
                const isNew     = newIds.has(id);
                const isSelected = selected === alert;
                return (
                  <>
                    <tr
                      key={i}
                      onClick={() => setSelected(isSelected ? null : alert)}
                      className={`cursor-pointer transition-all duration-150 ${isNew ? "animate-flash-new" : ""}`}
                      style={{
                        background:  isSelected
                          ? `${style.color}14`
                          : i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)",
                        borderLeft:  isSelected
                          ? `2px solid ${style.color}`
                          : "2px solid transparent",
                      }}
                      onMouseEnter={e => {
                        if (!isSelected)
                          (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.04)";
                      }}
                      onMouseLeave={e => {
                        if (!isSelected)
                          (e.currentTarget as HTMLElement).style.background =
                            i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.015)";
                      }}
                    >
                      <td
                        className="py-4 font-mono-code text-xs"
                        style={{ color: "rgba(255,255,255,0.35)", paddingLeft: "2rem" }}
                      >
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="py-4 px-4 font-mono-code text-sm" style={{ color: "#a1faff" }}>
                        {alert.src_ip}
                      </td>
                      <td className="py-4 px-4 text-sm font-medium" style={{ color: "#e8eafb" }}>
                        {alert.attack_type}
                      </td>
                      <td className="py-4 px-4">
                        <span
                          className="px-2.5 py-0.5 rounded text-[10px] font-black uppercase"
                          style={{
                            background: style.bg,
                            color:      style.color,
                            border:     `1px solid ${style.border}`,
                          }}
                        >
                          {style.label}
                        </span>
                      </td>
                      <td className="py-4 text-right" style={{ paddingRight: "2rem" }}>
                        <button
                          className="flex items-center gap-1 text-xs font-bold ml-auto transition-colors"
                          style={{ color: "#a1faff" }}
                          onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = "#fff"; }}
                          onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = "#a1faff"; }}
                        >
                          SHAP Explain
                          <ChevronRight size={14} />
                        </button>
                      </td>
                    </tr>
                    {isSelected && (
                      <tr key={`shap-${i}`}>
                        <td colSpan={5} className="px-8 pb-3">
                          <SHAPExplainer alert={alert} onClose={() => setSelected(null)} />
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
export default AlertFeed;
