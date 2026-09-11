import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  FileText,
  FileSpreadsheet,
  FileCode,
  Download,
} from "lucide-react";
import { getMissions, getCsvExportUrl, getJsonExportUrl, getPdfReportUrl } from "../api/missions";
import { getDashboardStats } from "../api/system";
import { Skeleton } from "../components/common/LoadingState";

export function Reports() {
  const [selectedMission] = useState<string>("MISSION-001");

  const { data: missions } = useQuery({
    queryKey: ["missions-reports"],
    queryFn: getMissions,
  });

  const { data: stats, isLoading: loadingStats } = useQuery({
    queryKey: ["dashboard-stats-reports"],
    queryFn: getDashboardStats,
  });

  return (
    <div className="flex flex-col gap-6">
      {/* Page Title */}
      <div>
        <h1 className="text-xl font-semibold text-text-navy">Mission Reports & Intelligence Export</h1>
        <p className="text-sm text-text-secondary">
          Generate official PDF survey dossiers, raw CSV logs, and JSON telemetry archives.
        </p>
      </div>

      {/* Summary Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <span className="text-xs font-semibold text-text-secondary uppercase">Surveys Analyzed</span>
          <div className="mt-2 text-2xl font-bold text-text-navy">
            {loadingStats ? <Skeleton className="h-8 w-16" /> : stats?.sonar_frames || 0}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <span className="text-xs font-semibold text-text-secondary uppercase">Identified Targets</span>
          <div className="mt-2 text-2xl font-bold text-color-ocean">
            {loadingStats ? <Skeleton className="h-8 w-16" /> : stats?.anomalies_detected || 0}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <span className="text-xs font-semibold text-text-secondary uppercase">High Priority Hazards</span>
          <div className="mt-2 text-2xl font-bold text-red-600">
            {loadingStats ? <Skeleton className="h-8 w-16" /> : stats?.high_priority || 0}
          </div>
        </div>

        <div className="rounded-xl border border-border bg-card p-4 shadow-xs">
          <span className="text-xs font-semibold text-text-secondary uppercase">Analysis Pipeline</span>
          <div className="mt-2 text-2xl font-bold text-emerald-600">Operational</div>
        </div>
      </div>

      {/* Export Action Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* PDF Dossier */}
        <div className="rounded-2xl border border-border bg-card p-6 shadow-xs flex flex-col justify-between">
          <div className="space-y-3">
            <div className="h-10 w-10 rounded-xl bg-red-50 border border-red-200 flex items-center justify-center text-red-600">
              <FileText className="h-5 w-5" />
            </div>
            <h3 className="text-base font-semibold text-text-navy">Executive PDF Report</h3>
            <p className="text-xs text-text-secondary">
              Includes sonar sonograph snapshots, bounding box telemetry, anomaly radar charts, and operator signatures.
            </p>
          </div>
          <div className="mt-6">
            <button
              onClick={() => window.open(getPdfReportUrl("AN-LATEST", selectedMission), "_blank")}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-color-ocean px-4 py-2.5 text-xs font-semibold text-white hover:bg-color-action transition-all shadow-xs cursor-pointer"
            >
              <Download className="h-4 w-4" /> Download Sonar PDF
            </button>
          </div>
        </div>

        {/* CSV Export */}
        <div className="rounded-2xl border border-border bg-card p-6 shadow-xs flex flex-col justify-between">
          <div className="space-y-3">
            <div className="h-10 w-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600">
              <FileSpreadsheet className="h-5 w-5" />
            </div>
            <h3 className="text-base font-semibold text-text-navy">Detection Telemetry CSV</h3>
            <p className="text-xs text-text-secondary">
              Structured spreadsheet containing all target classifications, confidence percentages, GPS coordinates, and depth readings.
            </p>
          </div>
          <div className="mt-6">
            <a
              href={getCsvExportUrl(selectedMission)}
              download
              className="w-full flex items-center justify-center gap-2 rounded-xl border border-border bg-bg-secondary px-4 py-2.5 text-xs font-semibold text-text-navy hover:bg-bg-primary transition-all shadow-xs"
            >
              <Download className="h-4 w-4" /> Export CSV Spreadsheet
            </a>
          </div>
        </div>

        {/* JSON Raw Archive */}
        <div className="rounded-2xl border border-border bg-card p-6 shadow-xs flex flex-col justify-between">
          <div className="space-y-3">
            <div className="h-10 w-10 rounded-xl bg-blue-50 border border-blue-200 flex items-center justify-center text-color-ocean">
              <FileCode className="h-5 w-5" />
            </div>
            <h3 className="text-base font-semibold text-text-navy">Full JSON Audit Dump</h3>
            <p className="text-xs text-text-secondary">
              Complete machine-readable JSON dataset for ingestion into naval GIS systems or secondary AI pipelines.
            </p>
          </div>
          <div className="mt-6">
            <a
              href={getJsonExportUrl(selectedMission)}
              download
              className="w-full flex items-center justify-center gap-2 rounded-xl border border-border bg-bg-secondary px-4 py-2.5 text-xs font-semibold text-text-navy hover:bg-bg-primary transition-all shadow-xs"
            >
              <Download className="h-4 w-4" /> Export JSON Archive
            </a>
          </div>
        </div>
      </div>

      {/* Available Missions Table */}
      <div className="rounded-2xl border border-border bg-card shadow-xs overflow-hidden">
        <div className="p-4 border-b border-border bg-bg-secondary flex items-center justify-between">
          <h2 className="text-sm font-semibold text-text-navy">Active Mission Records</h2>
          <span className="text-xs text-text-secondary">Select mission to filter export archives</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b border-border bg-bg-primary text-text-secondary font-semibold uppercase">
              <tr>
                <th className="px-4 py-3">Mission ID</th>
                <th className="px-4 py-3">Mission Name</th>
                <th className="px-4 py-3">Survey Area</th>
                <th className="px-4 py-3">Frames</th>
                <th className="px-4 py-3">Detections</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3 text-right">Quick Export</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-text-navy font-medium">
              {missions?.map((m) => (
                <tr key={m.mission_id} className="hover:bg-bg-primary/50 transition-colors">
                  <td className="px-4 py-3 font-mono font-semibold text-color-ocean">{m.mission_id}</td>
                  <td className="px-4 py-3">{m.name || "Standard Sonar Sweep"}</td>
                  <td className="px-4 py-3 text-text-secondary">{m.area || "Coastal Sector 4"}</td>
                  <td className="px-4 py-3">{m.image_count || 1}</td>
                  <td className="px-4 py-3 font-semibold">{m.detection_count || 0}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-600 border border-emerald-200">
                      Completed
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right space-x-2">
                    <a
                      href={getCsvExportUrl(m.mission_id)}
                      download
                      className="inline-block px-2.5 py-1 rounded border border-border bg-card text-text-navy hover:bg-bg-secondary font-medium"
                    >
                      CSV
                    </a>
                    <a
                      href={getJsonExportUrl(m.mission_id)}
                      download
                      className="inline-block px-2.5 py-1 rounded border border-border bg-card text-text-navy hover:bg-bg-secondary font-medium"
                    >
                      JSON
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
