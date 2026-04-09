import { useState } from "react";
import {
  LayoutDashboard, BellRing, FileText, Network,
  Brain, Settings, HelpCircle, BookOpen, Download,
  Shield,
} from "lucide-react";
const NAV_ITEMS = [
  { icon: LayoutDashboard, label: "Dashboard",       active: true  },
  { icon: BellRing,        label: "Alerts",          active: false },
  { icon: FileText,        label: "Reports",         active: false },
  { icon: Network,         label: "Network Map",     active: false },
  { icon: Brain,           label: "AI Explainability", active: false },
  { icon: Settings,        label: "Settings",        active: false },
];
const Sidebar = () => {
  const [active, setActive] = useState("Dashboard");
  return (
    <aside
      className="hidden lg:flex flex-col h-screen w-64 fixed left-0 top-0 z-50 border-r"
      style={{
        background: "linear-gradient(180deg, #0f131f 0%, #0a0e19 100%)",
        borderColor: "rgba(255,255,255,0.05)",
      }}
    >
      {}
      <div className="p-6 flex items-center gap-3 shrink-0">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center shadow-lg"
          style={{
            background: "linear-gradient(135deg, #a1faff, #699cff)",
            boxShadow: "0 4px 16px rgba(161,250,255,0.25)",
          }}
        >
          <Shield size={20} style={{ color: "#006165" }} />
        </div>
        <div>
          <h1
            className="font-bold text-xl leading-none"
            style={{ color: "#a1faff", fontFamily: "'Space Grotesk', sans-serif" }}
          >
            The Sentinel
          </h1>
          <p
            className="text-[10px] uppercase tracking-widest mt-1"
            style={{ color: "rgba(255,255,255,0.3)" }}
          >
            NIDS Command Center
          </p>
        </div>
      </div>
      {}
      <nav className="flex-1 px-4 py-2 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map(({ icon: Icon, label }) => {
          const isActive = active === label;
          return (
            <button
              key={label}
              onClick={() => setActive(label)}
              className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-all duration-150"
              style={{
                background: isActive ? "rgba(161,250,255,0.08)" : "transparent",
                borderLeft: isActive ? "3px solid #a1faff" : "3px solid transparent",
                color: isActive ? "#a1faff" : "rgba(255,255,255,0.35)",
              }}
              onMouseEnter={e => {
                if (!isActive) {
                  (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.7)";
                  (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.03)";
                }
              }}
              onMouseLeave={e => {
                if (!isActive) {
                  (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.35)";
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                }
              }}
            >
              <Icon size={18} />
              <span className="text-sm font-medium" style={{ fontFamily: "'Inter', sans-serif" }}>
                {label}
              </span>
            </button>
          );
        })}
      </nav>
      {}
      <div className="p-4 shrink-0 space-y-3" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
        <button
          className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg text-sm font-bold transition-all hover:brightness-110"
          style={{
            background: "rgba(161,250,255,0.15)",
            color: "#a1faff",
            border: "1px solid rgba(161,250,255,0.2)",
          }}
        >
          <Download size={16} />
          Export Logs
        </button>
        <div className="flex gap-4 px-1">
          <button
            className="flex items-center gap-1.5 text-xs transition-colors"
            style={{ color: "rgba(255,255,255,0.25)" }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.6)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.25)"; }}
          >
            <HelpCircle size={13} />
            <span>Support</span>
          </button>
          <button
            className="flex items-center gap-1.5 text-xs transition-colors"
            style={{ color: "rgba(255,255,255,0.25)" }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.6)"; }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = "rgba(255,255,255,0.25)"; }}
          >
            <BookOpen size={13} />
            <span>Docs</span>
          </button>
        </div>
      </div>
    </aside>
  );
};
export default Sidebar;
