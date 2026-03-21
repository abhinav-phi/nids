import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { X } from "lucide-react";
import type { Alert } from "@/hooks/useWebSocket";

interface Props {
  alert: Alert;
  onClose: () => void;
}

const SHAPExplainer = ({ alert, onClose }: Props) => {
  const shapData = (alert.shap_top5 || []).map((s) => ({
    feature: s.feature,
    value: s.value,
  }));

  return (
    <div className="animate-slide-down bg-card border border-border rounded-lg p-4 mt-2">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">
          Why was this flagged? — SHAP Feature Explanation
        </h3>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
          <X className="w-4 h-4" />
        </button>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4 text-xs">
        <div>
          <span className="text-muted-foreground">Prediction: </span>
          <span className="text-foreground font-medium">{alert.prediction || alert.attack_type}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Confidence: </span>
          <span className="text-foreground font-medium">{(alert.confidence * 100).toFixed(1)}%</span>
        </div>
        <div>
          <span className="text-muted-foreground">Severity: </span>
          <span className="font-medium">{alert.severity}</span>
        </div>
        <div>
          <span className="text-muted-foreground">Source IP: </span>
          <span className="font-mono-code text-foreground">{alert.src_ip}</span>
        </div>
      </div>
      {shapData.length === 0 ? (
        <div className="text-muted-foreground text-sm text-center py-4">
          No SHAP data available for this alert
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={shapData} layout="vertical" margin={{ left: 80 }}>
            <XAxis type="number" tick={{ fill: "#8b949e", fontSize: 10 }} />
            <YAxis
              type="category"
              dataKey="feature"
              tick={{ fill: "#8b949e", fontSize: 11 }}
              width={75}
            />
            <Tooltip contentStyle={{ backgroundColor: "#161b22", border: "1px solid #30363d", color: "#fff" }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {shapData.map((entry, i) => (
                <Cell key={i} fill={entry.value >= 0 ? "#f85149" : "#58a6ff"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default SHAPExplainer;
