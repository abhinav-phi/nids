import StatusBar from "@/components/StatusBar";
import KPICards from "@/components/KPICards";
import TrafficChart from "@/components/TrafficChart";
import AttackTimeline from "@/components/AttackTimeline";
import AttackPieChart from "@/components/AttackPieChart";
import AlertFeed from "@/components/AlertFeed";
import IPLeaderboard from "@/components/IPLeaderboard";
import { useWebSocket } from "@/hooks/useWebSocket";

const Index = () => {
  const { isConnected, alertHistory } = useWebSocket();

  return (
    <div className="min-h-screen bg-background">
      <StatusBar wsConnected={isConnected} />
      <main className="p-4 space-y-4 max-w-[1600px] mx-auto">
        <KPICards />
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 space-y-4">
            <TrafficChart alertHistory={alertHistory} />
            <AttackTimeline />
          </div>
          <div className="lg:col-span-2">
            <AlertFeed alertHistory={alertHistory} />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <AttackPieChart />
          <IPLeaderboard />
        </div>
      </main>
    </div>
  );
};

export default Index;
