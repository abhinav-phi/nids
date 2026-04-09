import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";
import { X, Brain } from "lucide-react";
import type { Alert } from "@/hooks/useWebSocket";
interface Props {
  alert: Alert;
  onClose: () => void;
}
const SHAPExplainer = ({ alert, onClose }: Props) => {
  const shapData = (alert.shap_top5 || []).map((s: any) => ({
    feature: s.feature,
    value:   s.value ?? s.impact ?? 0,
  }));
  const sev = (alert.severity || "LOW").toUpperCase();
  const sevColor = {
    CRITICAL: "#ff716c",
    HIGH:     "#699cff",
    MEDIUM:   "#ac8aff",
    LOW:      "rgba(255,255,255,0.4)",
  }[sev] || "#a1faff";
  return (
    <div
      className="animate-slide-down rounded-xl mt-2 p-5"
      style={{
        background:   "rgba(10,14,25,0.85)",
        border:       "1px solid rgba(161,250,255,0.15)",
        backdropFilter: "blur(12px)",
      }}
    >
      {}
      <div className="flex items-start justify-between mb-4">
        <div>
          <span
            className="inline-flex items-center gap-1.5 text-[10px] font-black uppercase tracking-widest px-2 py-0.5 rounded"
            style={{
              background: "rgba(161,250,255,0.08)",
              color:      "#a1faff",
              border:     "1px solid rgba(161,250,255,0.2)",
            }}
          >
            <Brain size={11} /> AI Insight
          </span>
          <h3
            className="text-base font-bold mt-1.5"
            style={{ color: "#e8eafb", fontFamily: "'Space Grotesk', sans-serif" }}
          >
            SHAP Explainer
          </h3>
          <p className="text-xs mt-0.5" style={{ color: "rgba(255,255,255,0.35)" }}>
            Why was this traffic flagged?
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-full transition-colors hover:bg-white/10"
          style={{ color: "rgba(255,255,255,0.4)" }}
        >
          <X size={16} />
        </button>
      </div>
      {}
      <div
        className="flex items-center gap-3 p-3 rounded-lg mb-4"
        style={{
          background: `${sevColor}10`,
          border:     `1px solid ${sevColor}30`,
        }}
      >
        <span className="text-sm font-bold" style={{ color: sevColor }}>
          Detection: {alert.attack_type || alert.prediction} — {(alert.confidence * 100).toFixed(1)}% confidence
        </span>
      </div>
      {}
      <div className="grid grid-cols-4 gap-2 mb-4">
        {[
          { label: "Prediction",  value: alert.attack_type || alert.prediction || "—" },
          { label: "Confidence",  value: `${(alert.confidence * 100).toFixed(1)}%` },
          { label: "Severity",    value: sev },
          { label: "Source IP",   value: alert.src_ip },
        ].map(({ label, value }) => (
          <div key={label}>
            <div className="text-[10px] uppercase tracking-widest mb-0.5"
              style={{ color: "rgba(255,255,255,0.3)" }}>{label}</div>
            <div className="text-xs font-medium font-mono-code"
              style={{ color: "#e8eafb" }}>{value}</div>
          </div>
        ))}
      </div>
      {}
      {shapData.length === 0 ? (
        <div className="text-center py-6 text-sm" style={{ color: "rgba(255,255,255,0.2)" }}>
          No SHAP data available for this alert
        </div>
      ) : (
        <div className="space-y-3">
          <p className="text-[10px] font-bold uppercase tracking-widest"
            style={{ color: "rgba(255,255,255,0.3)" }}>
            Top Contributing Features
          </p>
          {shapData.map((s) => (
            <div key={s.feature} className="space-y-1">
              <div className="flex justify-between text-xs font-mono-code">
                <span style={{ color: "#e8eafb" }}>{s.feature}</span>
                <span style={{ color: s.value >= 0 ? "#ff716c" : "#a1faff" }}>
                  {s.value >= 0 ? "+" : ""}{s.value.toFixed(3)}
                </span>
              </div>
              <div
                className="h-1.5 rounded-full overflow-hidden"
                style={{ background: "rgba(255,255,255,0.06)" }}
              >
                <div
                  className="h-full rounded-full transition-all"
                  style={{
                    width:      `${Math.min(Math.abs(s.value) * 200, 100)}%`,
                    background: s.value >= 0 ? "#ff716c" : "#a1faff",
                    marginLeft: s.value < 0 ? "auto" : undefined,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
export default SHAPExplainer;
