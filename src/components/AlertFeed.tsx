import { useState, useEffect, useRef } from "react";
import type { Alert } from "@/hooks/useWebSocket";
import SHAPExplainer from "./SHAPExplainer";

interface Props {
  alertHistory: Alert[];
}

const SEVERITY_BG: Record<string, string> = {
  CRITICAL: "rgba(248,81,73,0.15)",
  HIGH: "rgba(227,179,65,0.15)",
  MEDIUM: "rgba(63,185,80,0.15)",
  LOW: "rgba(139,148,158,0.1)",
};

const SEVERITY_COLOR: Record<string, string> = {
  CRITICAL: "#f85149",
  HIGH: "#e3b341",
  MEDIUM: "#3fb950",
  LOW: "#8b949e",
};

const AlertFeed = ({ alertHistory }: Props) => {
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const prevLenRef = useRef(0);

  const alerts = alertHistory.slice(0, 50);

  useEffect(() => {
    if (alertHistory.length > prevLenRef.current) {
      const newOnes = alertHistory.slice(0, alertHistory.length - prevLenRef.current);
      const ids = new Set(newOnes.map((a) => a.timestamp + a.src_ip));
      setNewIds(ids);
      const timer = setTimeout(() => setNewIds(new Set()), 500);
      prevLenRef.current = alertHistory.length;
      return () => clearTimeout(timer);
    }
  }, [alertHistory]);

  return (
    <div className="bg-card border border-border rounded-lg p-4 flex flex-col h-full">
      <h3 className="text-sm font-semibold text-foreground mb-3">Live Alert Feed</h3>
      {alerts.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
          No alerts yet
        </div>
      ) : (
        <div className="overflow-auto max-h-[400px] flex-1">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-card">
              <tr className="text-muted-foreground border-b border-border">
                <th className="text-left py-2 px-2">Time</th>
                <th className="text-left py-2 px-2">Source IP</th>
                <th className="text-left py-2 px-2">Attack Type</th>
                <th className="text-left py-2 px-2">Severity</th>
                <th className="text-left py-2 px-2">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map((alert, i) => {
                const sev = (alert.severity || "LOW").toUpperCase();
                const id = alert.timestamp + alert.src_ip;
                const isNew = newIds.has(id);
                return (
                  <tr
                    key={i}
                    onClick={() => setSelectedAlert(selectedAlert === alert ? null : alert)}
                    className={`cursor-pointer hover:bg-secondary/50 transition-colors ${isNew ? "animate-flash-new" : ""}`}
                    style={{ backgroundColor: isNew ? undefined : SEVERITY_BG[sev] || "transparent" }}
                  >
                    <td className="py-1.5 px-2 font-mono-code text-muted-foreground">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-1.5 px-2 font-mono-code text-foreground">{alert.src_ip}</td>
                    <td className="py-1.5 px-2 text-foreground">{alert.attack_type}</td>
                    <td className="py-1.5 px-2">
                      <span
                        className="px-2 py-0.5 rounded-full text-[10px] font-semibold"
                        style={{
                          backgroundColor: SEVERITY_BG[sev],
                          color: SEVERITY_COLOR[sev],
                        }}
                      >
                        {sev}
                      </span>
                    </td>
                    <td className="py-1.5 px-2">
                      <div className="flex items-center gap-1">
                        <div className="w-12 h-1.5 bg-secondary rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${alert.confidence * 100}%`,
                              backgroundColor: SEVERITY_COLOR[sev],
                            }}
                          />
                        </div>
                        <span className="text-muted-foreground">
                          {(alert.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {selectedAlert && (
        <SHAPExplainer alert={selectedAlert} onClose={() => setSelectedAlert(null)} />
      )}
    </div>
  );
};

export default AlertFeed;
