import StatusBar from "@/components/StatusBar";
import KPICards from "@/components/KPICards";
import TrafficChart from "@/components/TrafficChart";
import AttackTimeline from "@/components/AttackTimeline";
import AttackPieChart from "@/components/AttackPieChart";
import AlertFeed from "@/components/AlertFeed";
import IPLeaderboard from "@/components/IPLeaderboard";
import Sidebar from "@/components/Sidebar";
import { useWebSocket } from "@/hooks/useWebSocket";
const Index = () => {
  const { isConnected, alertHistory } = useWebSocket();
  return (
    <div className="min-h-screen bg-surface flex">
      {}
      <Sidebar />
      {}
      <div className="flex-1 lg:ml-64 flex flex-col min-h-screen">
        <StatusBar wsConnected={isConnected} />
        <main className="flex-1 p-6 space-y-6 max-w-[1600px] w-full mx-auto">
          {}
          <KPICards />
          {}
          <div className="grid grid-cols-12 gap-6">
            {}
            <div className="col-span-12 lg:col-span-8">
              <TrafficChart alertHistory={alertHistory} />
            </div>
            {}
            <div className="col-span-12 lg:col-span-4">
              <AttackPieChart />
            </div>
            {}
            <div className="col-span-12 lg:col-span-9" style={{ minHeight: 440 }}>
              <AlertFeed alertHistory={alertHistory} />
            </div>
            {}
            <div className="col-span-12 lg:col-span-3">
              <IPLeaderboard />
            </div>
            {}
            <div className="col-span-12">
              <AttackTimeline />
            </div>
          </div>
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
    </div>
  );
};
export default Index;
