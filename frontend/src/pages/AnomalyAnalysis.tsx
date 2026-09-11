import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  Radar,
  ShieldAlert,
  Sparkles,
} from "lucide-react";
import { ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import { getAllDetections } from "../api/missions";
import { Skeleton } from "../components/common/LoadingState";
import { ErrorState } from "../components/common/ErrorState";
import { EmptyState } from "../components/common/EmptyState";
import { StatusBadge } from "../components/common/StatusBadge";

const SEVERITY_COLORS = {
  HIGH: "#D9534F",
  MEDIUM: "#D99000",
  LOW: "#159A72",
  UNKNOWN: "#8884d8",
};

export function AnomalyAnalysis() {
  const [minAnomalyScore, setMinAnomalyScore] = useState<number>(0);

  const {
    data: detections,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["detections-anomaly"],
    queryFn: () => getAllDetections(undefined, 500),
  });

  const anomalyDetections = useMemo(() => {
    if (!detections) return [];
    return detections.filter(
      (d) => (d.is_anomaly || (d.anomaly_score ?? 0) > 0.3) && (d.anomaly_score ?? 0) >= minAnomalyScore
    );
  }, [detections, minAnomalyScore]);

  const severityData = useMemo(() => {
    if (!detections) return [];
    const counts = { HIGH: 0, MEDIUM: 0, LOW: 0 };
    detections.forEach((d) => {
      if (d.severity === "HIGH") counts.HIGH++;
      else if (d.severity === "MEDIUM") counts.MEDIUM++;
      else counts.LOW++;
    });
    return [
      { name: "High Risk", value: counts.HIGH, color: SEVERITY_COLORS.HIGH },
      { name: "Medium Risk", value: counts.MEDIUM, color: SEVERITY_COLORS.MEDIUM },
      { name: "Low Risk", value: counts.LOW, color: SEVERITY_COLORS.LOW },
    ].filter((item) => item.value > 0);
  }, [detections]);

  const classAnomalyData = useMemo(() => {
    if (!detections) return [];
    const map: Record<string, { name: string; anomalyCount: number; avgScore: number; total: number }> = {};
    detections.forEach((d) => {
      const cls = d.class_name.replace(/_/g, " ");
      if (!map[cls]) {
        map[cls] = { name: cls, anomalyCount: 0, avgScore: 0, total: 0 };
      }
      map[cls].total++;
      map[cls].avgScore += d.anomaly_score || 0;
      if (d.is_anomaly || (d.anomaly_score || 0) > 0.4) {
        map[cls].anomalyCount++;
      }
    });

    return Object.values(map).map((v) => ({
      name: v.name,
      anomalies: v.anomalyCount,
      avgScore: Math.round((v.avgScore / (v.total || 1)) * 100),
    }));
  }, [detections]);

  return (
    <div className="flex flex-col gap-6">
      {/* Page Title */}
      <div>
        <h1 className="text-xl font-semibold text-text-navy">Acoustic Shadow & Anomaly Analysis</h1>
        <p className="text-sm text-text-secondary">
          Deep acoustic shadow profiling, multi-sensor feature fusion, and automated hazard scoring.
        </p>
      </div>

      {/* Top Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <div className="flex items-center justify-between text-text-secondary">
            <span className="text-xs font-semibold uppercase">Total Anomalies</span>
            <AlertTriangle className="h-4 w-4 text-amber-500" />
          </div>
          <div className="mt-2 text-2xl font-bold text-text-navy">
            {isLoading ? <Skeleton className="h-8 w-16" /> : anomalyDetections.length}
          </div>
          <p className="mt-1 text-xs text-text-secondary">Acoustic signature outliers</p>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <div className="flex items-center justify-between text-text-secondary">
            <span className="text-xs font-semibold uppercase">High Risk Alerts</span>
            <ShieldAlert className="h-4 w-4 text-red-500" />
          </div>
          <div className="mt-2 text-2xl font-bold text-red-600">
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              detections?.filter((d) => d.severity === "HIGH").length || 0
            )}
          </div>
          <p className="mt-1 text-xs text-text-secondary">Requires immediate operator verification</p>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <div className="flex items-center justify-between text-text-secondary">
            <span className="text-xs font-semibold uppercase">Avg Shadow Index</span>
            <Radar className="h-4 w-4 text-color-ocean" />
          </div>
          <div className="mt-2 text-2xl font-bold text-text-navy">
            {isLoading ? (
              <Skeleton className="h-8 w-16" />
            ) : (
              `${Math.round(
                ((detections?.reduce((acc, d) => acc + (d.shadow_score || 0), 0) || 0) /
                  (detections?.length || 1)) *
                  100
              )}%`
            )}
          </div>
          <p className="mt-1 text-xs text-text-secondary">Shadow-to-highlight ratio</p>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <div className="flex items-center justify-between text-text-secondary">
            <span className="text-xs font-semibold uppercase">Fusion Confidence</span>
            <Sparkles className="h-4 w-4 text-teal-600" />
          </div>
          <div className="mt-2 text-2xl font-bold text-teal-600">92.4%</div>
          <p className="mt-1 text-xs text-text-secondary">YOLO + Shadow + Anomaly voting</p>
        </div>
      </div>

      {/* Chart Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Severity Breakdown */}
        <div className="rounded-xl border border-border bg-card p-5 shadow-xs flex flex-col">
          <h2 className="text-sm font-semibold text-text-navy mb-1">Risk Severity Distribution</h2>
          <p className="text-xs text-text-secondary mb-4">Proportion of targets classified by danger level.</p>
          <div className="h-64 w-full flex items-center justify-center">
            {severityData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={severityData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {severityData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="No detection data available for charts." />
            )}
          </div>
          <div className="flex justify-center gap-4 text-xs font-medium mt-2">
            {severityData.map((item) => (
              <div key={item.name} className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: item.color }} />
                <span className="text-text-navy">{item.name} ({item.value})</span>
              </div>
            ))}
          </div>
        </div>

        {/* Anomaly by Object Class */}
        <div className="rounded-xl border border-border bg-card p-5 shadow-xs flex flex-col">
          <h2 className="text-sm font-semibold text-text-navy mb-1">Anomaly Score by Target Class</h2>
          <p className="text-xs text-text-secondary mb-4">Mean anomaly deviation score across underwater objects.</p>
          <div className="h-64 w-full">
            {classAnomalyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={classAnomalyData} margin={{ top: 10, right: 20, left: -10, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E7EB" />
                  <XAxis dataKey="name" angle={-15} textAnchor="end" tick={{ fontSize: 10, fill: "#60758A" }} />
                  <YAxis unit="%" tick={{ fontSize: 10, fill: "#60758A" }} />
                  <Tooltip />
                  <Bar dataKey="avgScore" name="Avg Anomaly %" fill="#087EA4" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState message="No class distribution data available." />
            )}
          </div>
        </div>
      </div>

      {/* High Anomaly Flags List */}
      <div className="rounded-xl border border-border bg-card shadow-xs overflow-hidden">
        <div className="p-4 border-b border-border bg-bg-secondary flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-text-navy">Flagged Anomaly Targets</h3>
            <p className="text-xs text-text-secondary">Objects exhibiting abnormal acoustic shadows or shape divergence.</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary font-medium">Min Anomaly:</span>
            <input
              type="range"
              min="0"
              max="0.9"
              step="0.1"
              value={minAnomalyScore}
              onChange={(e) => setMinAnomalyScore(parseFloat(e.target.value))}
              className="w-24 accent-color-ocean"
            />
            <span className="text-xs font-semibold text-text-navy">{Math.round(minAnomalyScore * 100)}%</span>
          </div>
        </div>

        {isError ? (
          <div className="p-6">
            <ErrorState error={error} onRetry={() => refetch()} />
          </div>
        ) : isLoading ? (
          <div className="p-4 space-y-3">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : anomalyDetections.length === 0 ? (
          <div className="p-12 text-center">
            <EmptyState message="No high-anomaly objects detected matching the threshold criteria." />
          </div>
        ) : (
          <div className="divide-y divide-border">
            {anomalyDetections.map((d) => (
              <div key={d.detection_id} className="p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:bg-bg-primary/50 transition-colors">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-text-navy">{d.detection_id}</span>
                    <span className="px-2 py-0.5 rounded text-xs font-semibold bg-red-100 text-red-800">
                      Anomaly Score: {Math.round((d.anomaly_score || 0) * 100)}%
                    </span>
                    <StatusBadge kind="severity" value={d.severity} />
                  </div>
                  <div className="text-xs text-text-secondary">
                    Target: <span className="font-semibold text-text-navy">{d.class_name.replace(/_/g, " ")}</span> | Acoustic Shadow: {Math.round((d.shadow_score || 0) * 100)}% | Model Confidence: {Math.round(d.confidence * 100)}%
                  </div>
                  {d.lat !== null && d.lon !== null && (
                    <div className="text-xs text-text-secondary">
                      GPS: {d.lat?.toFixed(5)}, {d.lon?.toFixed(5)} {d.depth_m ? `| Depth: ${d.depth_m}m` : ""}
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  <span className="text-xs text-text-secondary italic">
                    {d.severity === "HIGH" ? "Immediate hazard - requires sonograph verification" : "Monitor in survey track"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
