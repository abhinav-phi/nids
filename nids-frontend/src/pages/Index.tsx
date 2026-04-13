import StatusBar from "@/components/StatusBar";
import KPICards from "@/components/KPICards";
import TrafficChart from "@/components/TrafficChart";
import AttackTimeline from "@/components/AttackTimeline";
import AttackPieChart from "@/components/AttackPieChart";
import AlertFeed from "@/components/AlertFeed";
import IPLeaderboard from "@/components/IPLeaderboard";
import Sidebar from "@/components/Sidebar";
<<<<<<< HEAD
import SHAPExplainer from "@/components/SHAPExplainer";
=======
import Chatbot from "@/components/Chatbot";
>>>>>>> 3fe5d089df572b708dfd118e3f1f1a167ab91b68
import { useWebSocket } from "@/hooks/useWebSocket";
import { useState } from "react";

const Index = () => {
  const { isConnected, alertHistory } = useWebSocket();
  const [activeTab, setActiveTab] = useState("Dashboard");
  return (
    <div className="min-h-screen bg-surface flex">
      {}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      {}
      <div className="flex-1 lg:ml-64 flex flex-col min-h-screen">
        <StatusBar wsConnected={isConnected} />
        <main className="flex-1 p-6 space-y-6 max-w-[1600px] w-full mx-auto">
          {activeTab === "Dashboard" && (
            <>
              <KPICards />
              <div className="grid grid-cols-12 gap-6">
                <div className="col-span-12 lg:col-span-8">
                  <TrafficChart alertHistory={alertHistory} />
                </div>
                <div className="col-span-12 lg:col-span-4">
                  <AttackPieChart />
                </div>
                <div className="col-span-12 lg:col-span-9" style={{ minHeight: 440 }}>
                  <AlertFeed alertHistory={alertHistory} />
                </div>
                <div className="col-span-12 lg:col-span-3">
                  <IPLeaderboard />
                </div>
                <div className="col-span-12">
                  <AttackTimeline />
                </div>
              </div>
            </>
          )}
          {activeTab === "Alerts" && (
            <div className="grid grid-cols-1 gap-6">
              <AlertFeed alertHistory={alertHistory} />
              <AttackTimeline />
            </div>
          )}
          {activeTab === "Reports" && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AttackPieChart />
              <IPLeaderboard />
            </div>
          )}
          {activeTab === "Network Map" && (
            <div className="grid grid-cols-1 gap-6">
              <TrafficChart alertHistory={alertHistory} />
            </div>
          )}
          {activeTab === "AI Explainability" && (
            <div className="grid grid-cols-1 gap-6 max-w-4xl mx-auto w-full">
               <h2 className="text-2xl font-bold font-headline mb-4">Latest Threat Explanation</h2>
               {alertHistory.length > 0 ? (
                 <SHAPExplainer alert={alertHistory[0]} onClose={() => setActiveTab("Dashboard")} />
               ) : (
                 <p className="opacity-50">Waiting for live alerts to provide AI explanation...</p>
               )}
            </div>
          )}
          {activeTab === "Settings" && (
            <div className="flex-1 flex flex-col items-center justify-center min-h-[60vh] opacity-50 space-y-4">
               <h2 className="text-3xl font-bold font-headline">Settings</h2>
               <p>System configuration requires Administrator privileges.</p>
            </div>
          )}
        </main>
        <footer className="border-t px-6 py-4 flex justify-between items-center"
          style={{ borderColor: "rgba(255,255,255,0.05)" }}>
          <span className="text-xs" style={{ color: "rgba(255,255,255,0.25)" }}>
            © 2024 The Sentinel — NIDS Command Center v1.0.0
          </span>
          <div className="flex gap-4 text-[10px] font-bold uppercase tracking-widest"
            style={{ color: "rgba(255,255,255,0.2)" }}>
            <span className="hover:text-cyan-400 cursor-pointer transition-colors">Privacy</span>
            <span className="hover:text-cyan-400 cursor-pointer transition-colors">System Logs</span>
          </div>
        </footer>
      </div>

      {/* Floating AI Chatbot */}
      <Chatbot />
    </div>
  );
};
export default Index;
