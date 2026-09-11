import { useQuery } from "@tanstack/react-query";
import { Waves, AlertTriangle, ShieldAlert, Gauge, Timer } from "lucide-react";
import { getDashboardStats } from "../api/system";
import { MetricCard } from "../components/common/MetricCard";
import { Skeleton } from "../components/common/LoadingState";
import { ErrorState } from "../components/common/ErrorState";
import { EmptyState } from "../components/common/EmptyState";

export function Dashboard() {
  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: getDashboardStats,
  });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-text-navy">Dashboard</h1>
        <p className="text-sm text-text-secondary">
          Mission overview and recent analysis activity.
        </p>
      </div>

      {isError ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <div className="grid grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24" />
          ))}
        </div>
      ) : data ? (
        <div className="grid grid-cols-5 gap-4">
          <MetricCard label="Sonar Frames" value={data.sonar_frames} icon={Waves} />
          <MetricCard
            label="Anomalies Detected"
            value={data.anomalies_detected}
            icon={AlertTriangle}
          />
          <MetricCard label="High Priority" value={data.high_priority} icon={ShieldAlert} />
          <MetricCard
            label="Average Confidence"
            value={
              data.average_confidence !== null
                ? `${data.average_confidence.toFixed(1)}%`
                : "Not available"
            }
            icon={Gauge}
          />
          <MetricCard
            label="Processing Time"
            value={
              data.average_processing_time_ms !== null
                ? `${data.average_processing_time_ms.toFixed(0)} ms`
                : "Not available"
            }
            icon={Timer}
          />
        </div>
      ) : null}

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 rounded-lg border border-border bg-card p-4">
          <h2 className="mb-3 text-sm font-semibold text-text-navy">
            Latest Sonar Analysis
          </h2>
          <EmptyState message="No analysis available yet. Run a sonar analysis to see the most recent result here." />
        </div>

        <div className="rounded-lg border border-border bg-card p-4">
          <h2 className="mb-3 text-sm font-semibold text-text-navy">
            Recent Detections
          </h2>
          <EmptyState message="No detections yet." />
        </div>
      </div>

      <div className="rounded-lg border border-border bg-card p-4">
        <h2 className="mb-3 text-sm font-semibold text-text-navy">
          Recent High-Priority Anomalies
        </h2>
        <EmptyState message="No high-priority anomalies recorded." />
      </div>
    </div>
  );
}
