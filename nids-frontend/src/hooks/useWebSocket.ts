import { useState, useEffect, useRef, useCallback } from "react";
export interface SHAPEntry {
  feature: string;
  value?:  number;    
  impact?: number;    
}
export interface Alert {
  id?:          string;
  timestamp:    string;
  src_ip:       string;
  source_ip?:   string;
  attack_type?: string;
  prediction?:  string;
  severity:     string;
  confidence:   number;
  shap_top5?:   SHAPEntry[];
  [key: string]: unknown;
}
function normalizeAlert(raw: Record<string, unknown>): Alert {
  return {
    ...raw,
    src_ip:      (raw.src_ip as string)      || (raw.source_ip as string) || "unknown",
    attack_type: (raw.attack_type as string) || (raw.prediction as string) || "Unknown",
    severity:    (raw.severity as string)    || "LOW",
    confidence:  (raw.confidence as number)  || 0,
    timestamp:   (raw.timestamp as string)   || new Date().toISOString(),
    shap_top5: ((raw.shap_top5 as any[]) || []).map((s: any) => ({
      feature: s.feature,
      value:   s.value ?? s.impact ?? 0,
      impact:  s.value ?? s.impact ?? 0,  
    })),
  } as Alert;
}
export function useWebSocket(url = "ws://localhost:8000/ws/live") {
  const [lastAlert, setLastAlert]       = useState<Alert | null>(null);
  const [isConnected, setIsConnected]   = useState(false);
  const [alertHistory, setAlertHistory] = useState<Alert[]>([]);
  const wsRef        = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();
  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => {
        console.log("[WS] Connected to NIDS backend");
        setIsConnected(true);
      };
      ws.onmessage = (event) => {
        try {
          const raw = JSON.parse(event.data);
          if (raw?.type === "ping") return;
          if (Array.isArray(raw)) {
            const normalized = raw.map(normalizeAlert);
            setAlertHistory(normalized.slice(0, 100));
          } else {
            const alert = normalizeAlert(raw);
            setLastAlert(alert);
            setAlertHistory((prev) => [alert, ...prev].slice(0, 100));
          }
        } catch (e) {
          console.error("[WS] Parse error", e);
        }
      };
      ws.onclose = () => {
        console.log("[WS] Disconnected — reconnecting in 3s...");
        setIsConnected(false);
        reconnectRef.current = setTimeout(connect, 3000);
      };
      ws.onerror = (err) => {
        console.error("[WS] Error", err);
        ws.close();
      };
    } catch (e) {
      console.error("[WS] Connection failed", e);
      reconnectRef.current = setTimeout(connect, 3000);
    }
  }, [url]);
  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [connect]);
  return { lastAlert, isConnected, alertHistory };
}
