import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./pages/Dashboard";
import { SonarAnalysis } from "./pages/SonarAnalysis";
import { Detections } from "./pages/Detections";
import { AnomalyAnalysis } from "./pages/AnomalyAnalysis";
import { MapView } from "./pages/MapView";
import { Reports } from "./pages/Reports";
import { ModelView } from "./pages/ModelView";
import { SystemStatus } from "./pages/SystemStatus";

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/sonar-analysis" element={<SonarAnalysis />} />
        <Route path="/detections" element={<Detections />} />
        <Route path="/anomaly-analysis" element={<AnomalyAnalysis />} />
        <Route path="/map" element={<MapView />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/model" element={<ModelView />} />
        <Route path="/system-status" element={<SystemStatus />} />
      </Routes>
    </AppShell>
  );
}

export default App;
