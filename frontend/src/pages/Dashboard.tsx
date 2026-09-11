import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Waves,
  AlertTriangle,
  ShieldAlert,
  Gauge,
  Timer,
  PlusCircle,
  ListTree,
  Map as MapIcon,
  FileText,
  ExternalLink,
  XCircle,
} from "lucide-react";
import { getDashboardStats } from "../api/system";
import { getLatestAnalysis, analysisImageUrl } from "../api/sonar";
import { getAllDetections, reviewDetection } from "../api/missions";
import type { DetectionRecord, AnalysisResponse, Severity } from "../api/types";
import { MetricCard } from "../components/common/MetricCard";
import { Skeleton } from "../components/common/LoadingState";
import { ErrorState } from "../components/common/ErrorState";
import { EmptyState } from "../components/common/EmptyState";
import { StatusBadge } from "../components/common/StatusBadge";

// Established taxonomy display mapping
const CLASS_LABELS: Record<string, string> = {
  crab_pot: "Crab Pot",
  submarine_pipeline: "Submarine Pipeline",
  shipwreck: "Shipwreck",
  ghost_net: "Ghost Net",
  mine_cylinder: "Mine Cylinder",
};

export function Dashboard() {
  const queryClient = useQueryClient();
  const [reviewModalDetection, setReviewModalDetection] = useState<DetectionRecord | null>(null);
  const [reviewStatus, setReviewStatus] = useState<string>("CONFIRMED");
  const [reviewNote, setReviewNote] = useState<string>("");

  // 1. Authoritative Dashboard Statistics
  const {
    data: stats,
    isLoading: loadingStats,
    isError: errorStats,
    error: errStats,
    refetch: refetchStats,
  } = useQuery({
    queryKey: ["dashboard-stats"],
    queryFn: () => getDashboardStats(),
    refetchInterval: 15_000,
  });

  // 2. Latest Analysis
  const {
    data: latestAnalysis,
    isLoading: loadingLatest,
  } = useQuery({
    queryKey: ["latest-analysis"],
    queryFn: getLatestAnalysis,
    refetchInterval: 15_000,
  });

  // 3. Recent Detections (Live from authoritative SQLite DB)
  const {
    data: recentDetections,
    isLoading: loadingDetections,
    isError: errorDetections,
    error: errDetections,
    refetch: refetchDetections,
  } = useQuery({
    queryKey: ["recent-detections"],
    queryFn: () => getAllDetections(undefined, 8),
    refetchInterval: 15_000,
  });

  // Mutation to persist operator review
  const reviewMutation = useMutation({
    mutationFn: ({ detectionId, status, note }: { detectionId: string; status: string; note: string }) =>
      reviewDetection(detectionId, { status, note }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
      queryClient.invalidateQueries({ queryKey: ["recent-detections"] });
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      queryClient.invalidateQueries({ queryKey: ["latest-analysis"] });
      setReviewModalDetection(null);
      setReviewNote("");
    },
  });

  const handleOpenReview = (detection: DetectionRecord) => {
    setReviewModalDetection(detection);
    setReviewStatus(detection.operator_status || "CONFIRMED");
    setReviewNote(detection.operator_note || "");
  };

  const handleSaveReview = () => {
    if (!reviewModalDetection) return;
    reviewMutation.mutate({
      detectionId: reviewModalDetection.detection_id,
      status: reviewStatus,
      note: reviewNote,
    });
  };

  // High-priority detections from live backend
  const highPriorityDetections = recentDetections?.filter((d) => d.severity === "HIGH") || [];

  // Determine highest severity of latest analysis
  const getLatestHighestSeverity = (analysis: AnalysisResponse): Severity => {
    if (!analysis.detections || analysis.detections.length === 0) return "LOW";
    if (analysis.detections.some((d) => d.severity === "HIGH")) return "HIGH";
    if (analysis.detections.some((d) => d.severity === "MEDIUM")) return "MEDIUM";
    return "LOW";
  };

  // Processing time formatting
  const getProcessingTimeDisplay = () => {
    if (latestAnalysis?.processing_time_ms) {
      return `${latestAnalysis.processing_time_ms.toFixed(0)} ms`;
    }
    if (stats?.average_processing_time_ms !== null && stats?.average_processing_time_ms !== undefined) {
      return `${stats.average_processing_time_ms.toFixed(1)} ms`;
    }
    return "Unavailable";
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Top Header & Quick Actions */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text-navy">Dashboard</h1>
          <p className="text-sm text-text-secondary">
            Mission overview and recent analysis activity.
          </p>
        </div>

        {/* Quick Action Navigation Bar */}
        <div className="flex flex-wrap items-center gap-2">
          <Link
            to="/sonar-analysis"
            className="flex items-center gap-1.5 rounded-lg bg-action px-3 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-action/90 transition-colors"
          >
            <PlusCircle className="h-3.5 w-3.5" />
            New Sonar Analysis
          </Link>
          <Link
            to="/detections"
            className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy shadow-xs hover:bg-bg-secondary transition-colors"
          >
            <ListTree className="h-3.5 w-3.5 text-ocean" />
            View Detections
          </Link>
          <Link
            to="/map"
            className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy shadow-xs hover:bg-bg-secondary transition-colors"
          >
            <MapIcon className="h-3.5 w-3.5 text-teal" />
            View Map
          </Link>
          <Link
            to="/reports"
            className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy shadow-xs hover:bg-bg-secondary transition-colors"
          >
            <FileText className="h-3.5 w-3.5 text-text-secondary" />
            Generate Report
          </Link>
        </div>
      </div>

      {/* 5 Authoritative Metric Cards */}
      {errorStats ? (
        <ErrorState error={errStats} onRetry={() => refetchStats()} />
      ) : loadingStats ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-24 rounded-xl" />
          ))}
        </div>
      ) : stats ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <MetricCard
            label="Sonar Frames"
            value={stats.sonar_frames}
            icon={Waves}
            className="shadow-xs"
          />
          <MetricCard
            label="Anomalies Detected"
            value={stats.anomalies_detected}
            icon={AlertTriangle}
            className="shadow-xs"
          />
          <MetricCard
            label="High Priority"
            value={stats.high_priority}
            icon={ShieldAlert}
            className="shadow-xs"
          />
          <MetricCard
            label="Average Confidence"
            value={
              stats.average_confidence !== null
                ? `${stats.average_confidence.toFixed(1)}%`
                : "Unavailable"
            }
            icon={Gauge}
            className="shadow-xs"
          />
          <MetricCard
            label="Processing Time"
            value={getProcessingTimeDisplay()}
            icon={Timer}
            className="shadow-xs"
          />
        </div>
      ) : null}

      {/* Two-Column Mid Section: Latest Sonar Analysis & Recent Detections */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Latest Sonar Analysis */}
        <div className="lg:col-span-2 rounded-xl border border-border bg-card p-5 shadow-xs flex flex-col justify-between">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-text-navy">
                Latest Sonar Analysis
              </h2>
              {latestAnalysis && (
                <span className="rounded bg-bg-secondary px-2 py-0.5 text-[11px] font-mono text-text-secondary border border-border">
                  {latestAnalysis.analysis_id}
                </span>
              )}
            </div>
            {latestAnalysis && (
              <Link
                to="/sonar-analysis"
                className="flex items-center gap-1 text-xs font-medium text-action hover:underline"
              >
                Open in Analysis Viewer
                <ExternalLink className="h-3 w-3" />
              </Link>
            )}
          </div>

          {loadingLatest ? (
            <div className="flex flex-col gap-3">
              <Skeleton className="h-44 w-full rounded-lg" />
              <div className="grid grid-cols-4 gap-2">
                <Skeleton className="h-12 rounded-md" />
                <Skeleton className="h-12 rounded-md" />
                <Skeleton className="h-12 rounded-md" />
                <Skeleton className="h-12 rounded-md" />
              </div>
            </div>
          ) : latestAnalysis ? (
            <div className="flex flex-col gap-4">
              {/* Image Preview & Details Grid */}
              <div className="flex flex-col md:flex-row gap-4">
                <div className="relative md:w-5/12 h-44 rounded-lg border border-border bg-black/90 overflow-hidden flex items-center justify-center">
                  <img
                    src={analysisImageUrl(latestAnalysis.analysis_id, "annotated")}
                    alt="Latest Sonar Analysis"
                    className="h-full w-full object-contain"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = analysisImageUrl(
                        latestAnalysis.analysis_id,
                        "raw"
                      );
                    }}
                  />
                  <div className="absolute bottom-2 left-2 rounded bg-black/70 px-2 py-0.5 text-[10px] text-white">
                    {latestAnalysis.image.width} × {latestAnalysis.image.height} px
                  </div>
                </div>

                {/* Metadata Breakdown */}
                <div className="md:w-7/12 grid grid-cols-2 gap-3 text-xs">
                  <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                    <span className="text-text-secondary font-medium block">Mission ID</span>
                    <span className="text-text-navy font-semibold text-sm">
                      {latestAnalysis.mission_id}
                    </span>
                  </div>

                  <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                    <span className="text-text-secondary font-medium block">Targets Identified</span>
                    <span className="text-text-navy font-semibold text-sm">
                      {latestAnalysis.detections.length} ({latestAnalysis.num_known} known)
                    </span>
                  </div>

                  <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                    <span className="text-text-secondary font-medium block">Highest Severity</span>
                    <div className="mt-0.5">
                      <StatusBadge kind="severity" value={getLatestHighestSeverity(latestAnalysis)} />
                    </div>
                  </div>

                  <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                    <span className="text-text-secondary font-medium block">Processing Latency</span>
                    <span className="text-text-navy font-semibold text-sm">
                      {latestAnalysis.processing_time_ms.toFixed(0)} ms
                    </span>
                  </div>

                  <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                    <span className="text-text-secondary font-medium block">Source Filename</span>
                    <span className="text-text-navy font-mono text-[11px] truncate block" title={latestAnalysis.image.filename}>
                      {latestAnalysis.image.filename}
                    </span>
                  </div>

                  <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                    <span className="text-text-secondary font-medium block">Location Reference</span>
                    <span className="text-text-navy font-semibold text-[11px]">
                      {latestAnalysis.detections[0]?.location_source || "UNAVAILABLE"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Detections Chips within Latest Analysis */}
              {latestAnalysis.detections.length > 0 ? (
                <div className="border-t border-border pt-3">
                  <span className="text-xs font-semibold text-text-secondary uppercase tracking-wider block mb-2">
                    Detections in this frame
                  </span>
                  <div className="flex flex-wrap gap-2">
                    {latestAnalysis.detections.map((d) => (
                      <div
                        key={d.detection_id}
                        className="flex items-center gap-2 rounded-lg border border-border bg-bg-primary px-2.5 py-1 text-xs"
                      >
                        <span className="font-semibold text-text-navy">
                          {d.display_name || CLASS_LABELS[d.class_name] || d.class_name}
                        </span>
                        <span className="text-text-secondary">
                          {(d.confidence * 100).toFixed(0)}%
                        </span>
                        <StatusBadge kind="severity" value={d.severity} />
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-xs text-text-secondary italic border-t border-border pt-2">
                  No discrete objects detected above threshold in this frame.
                </div>
              )}
            </div>
          ) : (
            <div className="py-6 flex flex-col items-center justify-center text-center gap-3">
              <EmptyState message="No analysis available yet. Run a sonar analysis to see the most recent result here." />
              <Link
                to="/sonar-analysis"
                className="rounded-lg bg-action px-3.5 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-action/90 transition-colors"
              >
                Run Sonar Analysis
              </Link>
            </div>
          )}
        </div>

        {/* Recent Detections List */}
        <div className="rounded-xl border border-border bg-card p-5 shadow-xs flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-text-navy">
              Recent Detections
            </h2>
            <Link
              to="/detections"
              className="text-xs font-medium text-action hover:underline"
            >
              View All ({stats?.anomalies_detected || 0})
            </Link>
          </div>

          {loadingDetections ? (
            <div className="flex flex-col gap-2.5">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-14 rounded-lg" />
              ))}
            </div>
          ) : errorDetections ? (
            <ErrorState error={errDetections} onRetry={() => refetchDetections()} />
          ) : recentDetections && recentDetections.length > 0 ? (
            <div className="flex flex-col gap-2 overflow-y-auto max-h-[360px] pr-1">
              {recentDetections.map((d) => (
                <div
                  key={d.detection_id}
                  onClick={() => handleOpenReview(d)}
                  className="flex items-center justify-between rounded-lg border border-border bg-bg-primary p-2.5 hover:bg-bg-secondary cursor-pointer transition-colors"
                >
                  <div className="flex flex-col">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-semibold text-text-navy">
                        {CLASS_LABELS[d.class_name] || d.class_name.replace(/_/g, " ")}
                      </span>
                      <span className="text-[10px] font-mono text-text-secondary">
                        {d.detection_id}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-[11px] text-text-secondary mt-0.5">
                      <span>Conf: {(d.confidence * 100).toFixed(0)}%</span>
                      {d.evidence_score !== undefined && d.evidence_score !== null && (
                        <span>Ev: {(d.evidence_score * 100).toFixed(0)}%</span>
                      )}
                      <span>{d.mission_id || "MISSION-001"}</span>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <StatusBadge kind="severity" value={d.severity} />
                    <button
                      type="button"
                      className="text-[11px] font-medium text-action hover:underline px-1"
                    >
                      Review
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <EmptyState message="No detections recorded yet." />
          )}
        </div>
      </div>

      {/* Bottom Section: Recent High-Priority Anomalies */}
      <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="h-4 w-4 text-status-high" />
            <h2 className="text-sm font-semibold text-text-navy">
              Recent High-Priority Anomalies
            </h2>
            <span className="rounded-full bg-status-high/10 px-2 py-0.5 text-xs font-semibold text-status-high border border-status-high/20">
              {stats?.high_priority || highPriorityDetections.length} High-Risk
            </span>
          </div>
          <Link
            to="/detections"
            className="text-xs font-medium text-action hover:underline"
          >
            Audit All High-Risk Targets
          </Link>
        </div>

        {loadingDetections ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-28 rounded-lg" />
            ))}
          </div>
        ) : highPriorityDetections.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {highPriorityDetections.slice(0, 6).map((d) => (
              <div
                key={d.detection_id}
                className="rounded-lg border border-status-high/30 bg-status-high/5 p-3.5 flex flex-col justify-between hover:bg-status-high/10 transition-colors"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <span className="text-xs font-bold text-text-navy block">
                        {CLASS_LABELS[d.class_name] || d.class_name.replace(/_/g, " ")}
                      </span>
                      <span className="text-[11px] font-mono text-text-secondary">
                        {d.detection_id} • {d.mission_id || "MISSION-001"}
                      </span>
                    </div>
                    <StatusBadge kind="severity" value="HIGH" />
                  </div>

                  <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
                    <div className="text-text-secondary">
                      YOLO Conf:{" "}
                      <strong className="text-text-navy">
                        {(d.confidence * 100).toFixed(0)}%
                      </strong>
                    </div>
                    <div className="text-text-secondary">
                      Evidence:{" "}
                      <strong className="text-text-navy">
                        {d.evidence_score !== undefined && d.evidence_score !== null
                          ? `${(d.evidence_score * 100).toFixed(0)}%`
                          : "N/A"}
                      </strong>
                    </div>
                  </div>
                </div>

                <div className="mt-3 pt-2.5 border-t border-status-high/20 flex items-center justify-between">
                  <span className="text-[10px] text-text-secondary uppercase">
                    Status: <strong className="text-text-navy">{d.operator_status || "Pending"}</strong>
                  </span>
                  <button
                    type="button"
                    onClick={() => handleOpenReview(d)}
                    className="rounded bg-white px-2.5 py-1 text-xs font-semibold text-status-high border border-status-high/40 hover:bg-status-high hover:text-white transition-colors shadow-2xs"
                  >
                    Review Detection
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <EmptyState message="No high-priority anomalies recorded." />
        )}
      </div>

      {/* Operator Review & Explainability Modal */}
      {reviewModalDetection && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-xs animate-[fadeIn_150ms_ease-out]">
          <div className="w-full max-w-lg rounded-xl border border-border bg-card p-6 shadow-xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-border pb-3">
              <div>
                <h3 className="text-base font-semibold text-text-navy">
                  Detection Review & Explainability
                </h3>
                <span className="text-xs font-mono text-text-secondary">
                  {reviewModalDetection.detection_id} • {reviewModalDetection.mission_id || "MISSION-001"}
                </span>
              </div>
              <button
                type="button"
                onClick={() => setReviewModalDetection(null)}
                className="rounded-md p-1 text-text-secondary hover:bg-bg-secondary transition-colors"
              >
                <XCircle className="h-5 w-5" />
              </button>
            </div>

            {/* Explainability Breakdown */}
            <div className="my-4 grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">Object Class</span>
                <span className="text-text-navy font-bold text-sm">
                  {CLASS_LABELS[reviewModalDetection.class_name] || reviewModalDetection.class_name}
                </span>
              </div>

              <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">Assigned Severity</span>
                <div className="mt-0.5">
                  <StatusBadge kind="severity" value={reviewModalDetection.severity} />
                </div>
              </div>

              <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">YOLO Detection Confidence</span>
                <span className="text-text-navy font-bold text-sm">
                  {(reviewModalDetection.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">Fused Evidence Score</span>
                <span className="text-text-navy font-bold text-sm">
                  {reviewModalDetection.evidence_score !== undefined && reviewModalDetection.evidence_score !== null
                    ? `${(reviewModalDetection.evidence_score * 100).toFixed(1)}%`
                    : "N/A — not provided by backend"}
                </span>
              </div>

              <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">Acoustic Shadow Score</span>
                <span className="text-text-navy font-semibold text-xs">
                  {reviewModalDetection.shadow_score !== undefined && reviewModalDetection.shadow_score !== null
                    ? `${(reviewModalDetection.shadow_score * 100).toFixed(1)}%`
                    : "N/A — not provided by backend"}
                </span>
              </div>

              <div className="rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">Anomaly Autoencoder Loss</span>
                <span className="text-text-navy font-semibold text-xs">
                  {reviewModalDetection.anomaly_score !== undefined && reviewModalDetection.anomaly_score !== null
                    ? reviewModalDetection.anomaly_score.toFixed(4)
                    : "N/A — not provided by backend"}
                </span>
              </div>

              <div className="col-span-2 rounded-lg bg-bg-primary p-2.5 border border-border">
                <span className="text-text-secondary font-medium block">Bounding Box & Area</span>
                <span className="text-text-navy font-mono text-[11px]">
                  {reviewModalDetection.bbox && reviewModalDetection.bbox.length === 4
                    ? `[${reviewModalDetection.bbox.join(", ")}] • Dimensions: ${Math.round(reviewModalDetection.bbox[2] - reviewModalDetection.bbox[0])} × ${Math.round(reviewModalDetection.bbox[3] - reviewModalDetection.bbox[1])} px`
                    : "N/A — not provided by backend"}
                </span>
              </div>
            </div>

            {/* Operator Review Form */}
            <div className="border-t border-border pt-3">
              <label className="block text-xs font-semibold text-text-navy mb-1.5">
                Operator Assessment Decision:
              </label>
              <div className="flex gap-2 mb-3">
                {[
                  { label: "Confirmed", value: "CONFIRMED" },
                  { label: "False Alarm", value: "FALSE_ALARM" },
                  { label: "Rejected", value: "REJECTED" },
                ].map((s) => (
                  <button
                    key={s.value}
                    type="button"
                    onClick={() => setReviewStatus(s.value)}
                    className={`flex-1 rounded-lg border py-1.5 text-xs font-semibold transition-colors ${
                      reviewStatus === s.value
                        ? "border-ocean bg-ocean/10 text-ocean shadow-2xs"
                        : "border-border bg-bg-primary text-text-secondary hover:bg-bg-secondary"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              <label className="block text-xs font-semibold text-text-navy mb-1.5">
                Operator Mission Notes:
              </label>
              <textarea
                rows={3}
                value={reviewNote}
                onChange={(e) => setReviewNote(e.target.value)}
                placeholder="Enter verification notes, seafloor context, or action instructions..."
                className="w-full rounded-lg border border-border bg-bg-primary p-2.5 text-xs text-text-navy focus:outline-none focus:ring-1 focus:ring-ocean"
              />
            </div>

            {/* Action Buttons */}
            <div className="mt-4 flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setReviewModalDetection(null)}
                className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSaveReview}
                disabled={reviewMutation.isPending}
                className="rounded-lg bg-action px-4 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-action/90 disabled:opacity-50 transition-colors"
              >
                {reviewMutation.isPending ? "Saving..." : "Save Operator Feedback"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
