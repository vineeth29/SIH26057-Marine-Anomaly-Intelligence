import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./pages/Dashboard";
import { SonarAnalysis } from "./pages/SonarAnalysis";
import { ComingSoon } from "./pages/ComingSoon";

function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/sonar-analysis" element={<SonarAnalysis />} />
        <Route path="/detections" element={<ComingSoon pageName="Detections" />} />
        <Route path="/anomaly-analysis" element={<ComingSoon pageName="Anomaly Analysis" />} />
        <Route path="/map" element={<ComingSoon pageName="Map" />} />
        <Route path="/reports" element={<ComingSoon pageName="Reports" />} />
        <Route path="/model" element={<ComingSoon pageName="Model" />} />
        <Route path="/system-status" element={<ComingSoon pageName="System Status" />} />
      </Routes>
    </AppShell>
  );
}

export default App;
