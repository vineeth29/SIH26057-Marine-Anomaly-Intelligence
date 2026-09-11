import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Search,
  CheckCircle2,
  XCircle,
  AlertCircle,
  HelpCircle,
  FileSpreadsheet,
  FileCode,
} from "lucide-react";
import { getAllDetections, getMissions, reviewDetection, getCsvExportUrl, getJsonExportUrl } from "../api/missions";
import type { DetectionRecord } from "../api/types";
import { Skeleton } from "../components/common/LoadingState";
import { ErrorState } from "../components/common/ErrorState";
import { EmptyState } from "../components/common/EmptyState";
import { StatusBadge } from "../components/common/StatusBadge";

export function Detections() {
  const queryClient = useQueryClient();
  const [selectedMission, setSelectedMission] = useState<string>("");
  const [selectedClass, setSelectedClass] = useState<string>("ALL");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [reviewModalDetection, setReviewModalDetection] = useState<DetectionRecord | null>(null);
  const [reviewStatus, setReviewStatus] = useState<string>("CONFIRMED");
  const [reviewNote, setReviewNote] = useState<string>("");

  const { data: missions } = useQuery({
    queryKey: ["missions"],
    queryFn: getMissions,
  });

  const {
    data: detections,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["detections", selectedMission],
    queryFn: () => getAllDetections(selectedMission || undefined, 500),
  });

  const reviewMutation = useMutation({
    mutationFn: ({ detectionId, status, note }: { detectionId: string; status: string; note: string }) =>
      reviewDetection(detectionId, { status, note }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setReviewModalDetection(null);
      setReviewNote("");
    },
  });

  const filteredDetections = useMemo(() => {
    if (!detections) return [];
    return detections.filter((d) => {
      if (selectedClass !== "ALL" && d.class_name !== selectedClass) return false;
      if (selectedSeverity !== "ALL" && d.severity !== selectedSeverity) return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchesId = d.detection_id.toLowerCase().includes(q);
        const matchesClass = d.class_name.toLowerCase().includes(q);
        const matchesNote = d.operator_note?.toLowerCase().includes(q);
        if (!matchesId && !matchesClass && !matchesNote) return false;
      }
      return true;
    });
  }, [detections, selectedClass, selectedSeverity, searchQuery]);

  const uniqueClasses = useMemo(() => {
    if (!detections) return [];
    return Array.from(new Set(detections.map((d) => d.class_name))).filter(Boolean);
  }, [detections]);

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

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-text-navy">Detected Sonar Targets</h1>
          <p className="text-sm text-text-secondary">
            Verified objects, underwater hazards, pipelines, and operator review audit log.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <a
            href={getCsvExportUrl(selectedMission || undefined)}
            download
            className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy hover:bg-bg-secondary transition-colors shadow-xs"
          >
            <FileSpreadsheet className="h-4 w-4 text-color-ocean" />
            Export CSV
          </a>
          <a
            href={getJsonExportUrl(selectedMission || undefined)}
            download
            className="flex items-center gap-1.5 rounded-lg border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy hover:bg-bg-secondary transition-colors shadow-xs"
          >
            <FileCode className="h-4 w-4 text-teal-600" />
            Export JSON
          </a>
        </div>
      </div>

      {/* Filters Bar */}
      <div className="rounded-xl border border-border bg-card p-4 shadow-xs flex flex-wrap gap-4 items-center justify-between">
        <div className="flex flex-wrap items-center gap-3">
          {/* Mission Filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-text-secondary">Mission:</span>
            <select
              value={selectedMission}
              onChange={(e) => setSelectedMission(e.target.value)}
              className="rounded-md border border-border bg-bg-primary px-2.5 py-1.5 text-xs text-text-navy focus:outline-none focus:ring-1 focus:ring-color-ocean"
            >
              <option value="">All Missions</option>
              {missions?.map((m) => (
                <option key={m.mission_id} value={m.mission_id}>
                  {m.name || m.mission_id}
                </option>
              ))}
            </select>
          </div>

          {/* Class Filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-text-secondary">Class:</span>
            <select
              value={selectedClass}
              onChange={(e) => setSelectedClass(e.target.value)}
              className="rounded-md border border-border bg-bg-primary px-2.5 py-1.5 text-xs text-text-navy focus:outline-none focus:ring-1 focus:ring-color-ocean"
            >
              <option value="ALL">All Classes</option>
              {uniqueClasses.map((cls) => (
                <option key={cls} value={cls}>
                  {cls.replace(/_/g, " ").toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {/* Severity Filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-text-secondary">Severity:</span>
            <select
              value={selectedSeverity}
              onChange={(e) => setSelectedSeverity(e.target.value)}
              className="rounded-md border border-border bg-bg-primary px-2.5 py-1.5 text-xs text-text-navy focus:outline-none focus:ring-1 focus:ring-color-ocean"
            >
              <option value="ALL">All Severities</option>
              <option value="HIGH">High Risk</option>
              <option value="MEDIUM">Medium Risk</option>
              <option value="LOW">Low Risk</option>
            </select>
          </div>
        </div>

        {/* Search */}
        <div className="relative min-w-[220px]">
          <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-text-secondary" />
          <input
            type="text"
            placeholder="Search by ID, class, note..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-md border border-border bg-bg-primary pl-8 pr-3 py-1.5 text-xs text-text-navy placeholder:text-text-secondary focus:outline-none focus:ring-1 focus:ring-color-ocean"
          />
        </div>
      </div>

      {/* Table Section */}
      {isError ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <div className="space-y-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-14 w-full rounded-lg" />
          ))}
        </div>
      ) : filteredDetections.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-12 text-center">
          <EmptyState message="No sonar detections match your active filters." />
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-border bg-card shadow-xs">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-border bg-bg-secondary text-text-secondary uppercase tracking-wider font-semibold">
                <tr>
                  <th className="px-4 py-3">Detection ID</th>
                  <th className="px-4 py-3">Target Class</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Anomaly Score</th>
                  <th className="px-4 py-3">Shadow Score</th>
                  <th className="px-4 py-3">Severity</th>
                  <th className="px-4 py-3">Coordinates / Depth</th>
                  <th className="px-4 py-3">Operator Status</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border font-medium text-text-navy">
                {filteredDetections.map((d) => {
                  const confPct = Math.round(d.confidence * 100);
                  const anomPct = Math.round((d.anomaly_score || 0) * 100);
                  const shadowPct = Math.round((d.shadow_score || 0) * 100);

                  return (
                    <tr key={d.detection_id} className="hover:bg-bg-primary/60 transition-colors">
                      <td className="px-4 py-3 font-mono text-xs text-text-navy">
                        {d.detection_id}
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-50 text-color-ocean border border-blue-200">
                          {d.class_name.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-14 bg-gray-200 rounded-full h-1.5 overflow-hidden">
                            <div
                              className="bg-color-ocean h-1.5 rounded-full"
                              style={{ width: `${confPct}%` }}
                            />
                          </div>
                          <span>{confPct}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={anomPct > 60 ? "text-red-600 font-bold" : "text-text-secondary"}>
                          {anomPct}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-text-secondary">
                        {shadowPct}%
                      </td>
                      <td className="px-4 py-3">
                        <StatusBadge kind="severity" value={d.severity} />
                      </td>
                      <td className="px-4 py-3 text-text-secondary">
                        {d.lat !== null && d.lon !== null ? (
                          <span>{d.lat?.toFixed(4)}, {d.lon?.toFixed(4)} {d.depth_m ? `(${d.depth_m}m)` : ""}</span>
                        ) : (
                          <span className="italic text-gray-400">N/A</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        {d.operator_status === "CONFIRMED" ? (
                          <span className="inline-flex items-center gap-1 text-emerald-600 font-semibold text-xs">
                            <CheckCircle2 className="h-3.5 w-3.5" /> Confirmed
                          </span>
                        ) : d.operator_status === "REJECTED" ? (
                          <span className="inline-flex items-center gap-1 text-red-500 font-semibold text-xs">
                            <XCircle className="h-3.5 w-3.5" /> Rejected
                          </span>
                        ) : d.operator_status === "FALSE_ALARM" ? (
                          <span className="inline-flex items-center gap-1 text-amber-500 font-semibold text-xs">
                            <AlertCircle className="h-3.5 w-3.5" /> False Alarm
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-gray-400 text-xs">
                            <HelpCircle className="h-3.5 w-3.5" /> Pending
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleOpenReview(d)}
                          className="rounded-md border border-border bg-card px-2.5 py-1 text-xs font-medium text-text-navy hover:bg-bg-secondary hover:text-color-ocean transition-colors shadow-2xs cursor-pointer"
                        >
                          Review
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="border-t border-border bg-bg-secondary px-4 py-2.5 text-xs text-text-secondary flex justify-between items-center">
            <span>Showing {filteredDetections.length} total detections</span>
          </div>
        </div>
      )}

      {/* Operator Review Modal */}
      {reviewModalDetection && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
          <div className="w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-xl">
            <h3 className="text-base font-semibold text-text-navy mb-1">
              Review Detection #{reviewModalDetection.detection_id}
            </h3>
            <p className="text-xs text-text-secondary mb-4">
              Class: <span className="font-semibold text-text-navy">{reviewModalDetection.class_name}</span> | Severity: {reviewModalDetection.severity}
            </p>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-text-navy mb-1.5">
                  Operator Decision
                </label>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { id: "CONFIRMED", label: "Confirmed", icon: CheckCircle2, color: "text-emerald-600" },
                    { id: "FALSE_ALARM", label: "False Alarm", icon: AlertCircle, color: "text-amber-500" },
                    { id: "REJECTED", label: "Rejected", icon: XCircle, color: "text-red-500" },
                  ].map((opt) => {
                    const Icon = opt.icon;
                    return (
                      <button
                        key={opt.id}
                        type="button"
                        onClick={() => setReviewStatus(opt.id)}
                        className={`flex flex-col items-center justify-center p-2.5 rounded-lg border text-xs font-medium transition-all cursor-pointer ${
                          reviewStatus === opt.id
                            ? "border-color-ocean bg-blue-50/50 text-color-ocean ring-1 ring-color-ocean"
                            : "border-border bg-card text-text-secondary hover:bg-bg-primary"
                        }`}
                      >
                        <Icon className={`h-4 w-4 mb-1 ${opt.color}`} />
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-text-navy mb-1.5">
                  Operator Notes / Remarks
                </label>
                <textarea
                  rows={3}
                  value={reviewNote}
                  onChange={(e) => setReviewNote(e.target.value)}
                  placeholder="Add inspection notes or sonar feature explanations..."
                  className="w-full rounded-lg border border-border bg-bg-primary p-2.5 text-xs text-text-navy placeholder:text-text-secondary focus:outline-none focus:ring-1 focus:ring-color-ocean"
                />
              </div>
            </div>

            <div className="mt-6 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setReviewModalDetection(null)}
                className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-bg-secondary cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={reviewMutation.isPending}
                onClick={handleSaveReview}
                className="rounded-lg bg-color-ocean px-4 py-1.5 text-xs font-medium text-white hover:bg-color-action transition-colors disabled:opacity-50 cursor-pointer"
              >
                {reviewMutation.isPending ? "Saving..." : "Save Feedback"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
