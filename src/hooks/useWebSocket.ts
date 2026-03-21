import { useState, useEffect, useRef, useCallback } from "react";

export interface Alert {
  id?: string;
  timestamp: string;
  src_ip: string;
  attack_type: string;
  severity: string;
  confidence: number;
  prediction?: string;
  shap_top5?: { feature: string; value: number }[];
  [key: string]: unknown;
}

export function useWebSocket(url = "ws://localhost:8000/ws/live") {
  const [lastAlert, setLastAlert] = useState<Alert | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [alertHistory, setAlertHistory] = useState<Alert[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("[WS] Connected");
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const alert: Alert = JSON.parse(event.data);
          setLastAlert(alert);
          setAlertHistory((prev) => [alert, ...prev].slice(0, 100));
        } catch (e) {
          console.error("[WS] Parse error", e);
        }
      };

      ws.onclose = () => {
        console.log("[WS] Disconnected, reconnecting in 3s...");
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
