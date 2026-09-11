import { useState } from "react";
import { ChevronDown, ChevronUp, Shield, Sliders, CheckCircle, Info } from "lucide-react";
import type { Detection } from "../../api/types";
import { StatusBadge } from "../common/StatusBadge";
import { EmptyState } from "../common/EmptyState";

interface DetectionListProps {
  detections: Detection[];
  quality?: Record<string, unknown>;
}

export function DetectionList({ detections, quality }: DetectionListProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  if (detections.length === 0) {
    return <EmptyState message="No anomalies were detected in this sonar image." />;
  }

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const expandAll = () => {
    setExpandedIds(new Set(detections.map((d) => d.detection_id)));
  };

  const collapseAll = () => {
    setExpandedIds(new Set());
  };

  // Extract actual image quality summary if available
  const qualityGrade = (quality as any)?.quality || (quality as any)?.grade || null;
  const qualityScore = (quality as any)?.score !== undefined
    ? `${Math.round(Number((quality as any).score) * 100)}%`
    : (quality as any)?.quality_score !== undefined
    ? `${Math.round(Number((quality as any).quality_score) * 100)}%`
    : null;
  const qualityDisplay = qualityGrade
    ? `${qualityGrade}${qualityScore ? ` (${qualityScore})` : ""}`
    : "N/A — not provided by backend";

  return (
    <div className="flex flex-col gap-3">
      {/* Expand/Collapse Header Bar */}
      <div className="flex items-center justify-between pb-1 text-xs text-text-secondary">
        <span>{detections.length} target{detections.length === 1 ? "" : "s"} identified</span>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={expandAll}
            className="text-color-ocean hover:underline font-medium cursor-pointer"
          >
            Expand All
          </button>
          <span>•</span>
          <button
            type="button"
            onClick={collapseAll}
            className="text-text-secondary hover:underline cursor-pointer"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Detection Cards List with Subtle Staggered Entrance */}
      <div className="flex flex-col gap-2.5">
        {detections.map((d, index) => {
          const isExpanded = expandedIds.has(d.detection_id);
          const isHighSeverity = d.severity === "HIGH";

          // Calculate bounding box metrics from actual values
          const bbox = d.bbox || [];
          const x1 = bbox[0] !== undefined ? Math.round(bbox[0]) : 0;
          const y1 = bbox[1] !== undefined ? Math.round(bbox[1]) : 0;
          const x2 = bbox[2] !== undefined ? Math.round(bbox[2]) : 0;
          const y2 = bbox[3] !== undefined ? Math.round(bbox[3]) : 0;
          const w = d.bbox_width_px || Math.max(0, x2 - x1);
          const h = d.bbox_height_px || Math.max(0, y2 - y1);
          const area = d.object_area_px || w * h;
          const aspectRatio = h > 0 ? (w / h).toFixed(2) : "N/A";

          // Dynamic Why / Description generated from actual confidence & classification
          let dynamicWhy = "";
          if (d.confidence >= 0.6) {
            dynamicWhy = `YOLO detector identified a possible ${d.display_name.toLowerCase()} candidate with high detector confidence (${d.confidence_pct.toFixed(0)}%), and the target was retained after downstream evidence and shadow analysis.`;
          } else if (d.confidence >= 0.4) {
            dynamicWhy = `YOLO detector identified a possible ${d.display_name.toLowerCase()} candidate with moderate detector confidence (${d.confidence_pct.toFixed(0)}%), and the target was retained after downstream evidence and false-positive verification.`;
          } else {
            dynamicWhy = `Lower-confidence ${d.display_name.toLowerCase()} candidate (${d.confidence_pct.toFixed(0)}% YOLO confidence) retained because it met downstream acoustic evidence criteria.`;
          }

          // Severity justification based on actual backend logic
          let severityWhy = "";
          if (d.severity === "HIGH") {
            severityWhy = `High-risk classification assigned based on target class profile (${d.class_name}) and combined evidence score (${(d.evidence_score * 100).toFixed(0)}%).`;
          } else if (d.severity === "MEDIUM") {
            severityWhy = `Medium-risk classification assigned based on target characteristics and downstream evidence scoring.`;
          } else {
            severityWhy = `Low-risk classification assigned based on standard object profile and background baseline.`;
          }

          // Stagger animation delay up to first 8 items
          const staggerDelay = Math.min(index * 30, 240);

          return (
            <div
              key={d.detection_id}
              style={{ animationDelay: `${staggerDelay}ms` }}
              className={`rounded-xl border transition-all duration-200 overflow-hidden bg-card animate-[pageEntrance_150ms_ease-out_both] ${
                isHighSeverity ? "animate-[newHighRowHighlight_800ms_ease-out]" : ""
              } ${
                isExpanded
                  ? "border-color-ocean/40 shadow-sm ring-1 ring-color-ocean/20"
                  : "border-border hover:border-border/80 hover:bg-bg-primary/30"
              }`}
            >
              {/* Card Header (Collapsed Summary) */}
              <button
                type="button"
                onClick={() => toggleExpand(d.detection_id)}
                className="w-full flex items-center justify-between p-3.5 text-left cursor-pointer transition-colors"
              >
                <div className="flex items-center gap-3">
                  <span className="font-mono text-[11px] font-bold text-text-secondary bg-bg-secondary px-2 py-0.5 rounded">
                    {d.detection_id}
                  </span>
                  <div>
                    <div className="text-sm font-semibold text-text-navy flex items-center gap-2">
                      <span>{d.display_name}</span>
                    </div>
                    <div className="text-xs text-text-secondary mt-0.5">
                      <span className="font-medium text-color-ocean">YOLO Confidence: {d.confidence_pct.toFixed(1)}%</span>
                      {d.evidence_score !== undefined && (
                        <span className="ml-2 text-gray-400">| Evidence: {(d.evidence_score * 100).toFixed(0)}%</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <StatusBadge kind="severity" value={d.severity} />
                  <span
                    className={`text-[11px] font-bold px-2 py-0.5 rounded-full border ${
                      d.status === "RETAINED"
                        ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                        : "bg-gray-100 text-gray-700 border-gray-200"
                    }`}
                  >
                    {d.status}
                  </span>
                  <div className="p-1 text-text-secondary hover:text-text-navy">
                    {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </div>
                </div>
              </button>

              {/* Card Body (Expanded Detailed Parameters & Justifications) */}
              {isExpanded && (
                <div className="border-t border-border bg-bg-primary/40 p-4 space-y-4 text-xs animate-in fade-in-50 duration-150">
                  {/* 1. WHY / DESCRIPTION */}
                  <div className="rounded-lg bg-card border border-border p-3 space-y-1.5 shadow-2xs">
                    <div className="flex items-center gap-1.5 font-semibold text-text-navy">
                      <Info className="h-3.5 w-3.5 text-color-ocean" />
                      <span>Why / Description:</span>
                    </div>
                    <p className="text-text-navy leading-relaxed pl-5">
                      {dynamicWhy}
                    </p>
                    {d.reasons && d.reasons.length > 0 && (
                      <div className="pl-5 pt-1 space-y-0.5 text-text-secondary">
                        {d.reasons.map((r, i) => (
                          <div key={i} className="flex items-center gap-1.5">
                            <span className="h-1 w-1 rounded-full bg-color-ocean" />
                            <span>{r}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* 2. PARAMETERS GRID */}
                  <div className="rounded-lg bg-card border border-border p-3 space-y-2.5 shadow-2xs">
                    <div className="flex items-center gap-1.5 font-semibold text-text-navy">
                      <Sliders className="h-3.5 w-3.5 text-teal-600" />
                      <span>Actual Decision Parameters:</span>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1">
                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Class Name</div>
                        <div className="font-mono font-bold text-text-navy text-xs mt-0.5">{d.class_name}</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">YOLO Confidence</div>
                        <div className="font-bold text-color-ocean text-xs mt-0.5">{d.confidence_pct.toFixed(1)}%</div>
                        <div className="text-[9px] text-text-secondary mt-0.5">Detector confidence score</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Evidence Score</div>
                        <div className="font-bold text-text-navy text-xs mt-0.5">
                          {d.evidence_score !== undefined && d.evidence_score !== null
                            ? `${(d.evidence_score * 100).toFixed(1)}%`
                            : "N/A — not provided by backend"}
                        </div>
                        <div className="text-[9px] text-text-secondary mt-0.5">Multi-factor fused score</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Acoustic Shadow</div>
                        <div className="font-bold text-text-navy text-xs mt-0.5">
                          {d.shadow_score !== undefined && d.shadow_score !== null
                            ? `${(d.shadow_score * 100).toFixed(1)}%`
                            : "N/A — not provided by backend"}
                        </div>
                        <div className="text-[9px] text-text-secondary mt-0.5">Shadow-to-highlight ratio</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Anomaly Score</div>
                        <div className="font-bold text-text-navy text-xs mt-0.5">
                          {d.anomaly_score !== undefined && d.anomaly_score !== null
                            ? `${(d.anomaly_score * 100).toFixed(1)}%`
                            : "N/A — not provided by backend"}
                        </div>
                        <div className="text-[9px] text-text-secondary mt-0.5">Acoustic outlier index</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Bounding Box [X, Y, W, H]</div>
                        <div className="font-mono text-text-navy text-[11px] mt-0.5">
                          [{x1}, {y1}, {w}px, {h}px]
                        </div>
                        <div className="text-[9px] text-text-secondary mt-0.5">Area: {area} px² | Aspect: {aspectRatio}</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Image Quality</div>
                        <div className="font-semibold text-text-navy text-xs mt-0.5">{qualityDisplay}</div>
                        <div className="text-[9px] text-text-secondary mt-0.5">Frame signal assessment</div>
                      </div>

                      <div className="rounded border border-border/60 bg-bg-secondary/50 p-2">
                        <div className="text-[10px] uppercase font-semibold text-text-secondary">Filter Decision</div>
                        <div className="font-bold text-emerald-700 text-xs mt-0.5">{d.status}</div>
                        <div className="text-[9px] text-text-secondary mt-0.5">False-positive filter pass</div>
                      </div>
                    </div>
                  </div>

                  {/* 3. SEVERITY & FILTER DECISION BREAKDOWN */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {/* Severity Explanation */}
                    <div className="rounded-lg bg-card border border-border p-3 space-y-1.5 shadow-2xs">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-text-navy flex items-center gap-1.5">
                          <Shield className="h-3.5 w-3.5 text-red-500" />
                          Severity Assignment:
                        </span>
                        <StatusBadge kind="severity" value={d.severity} />
                      </div>
                      <p className="text-text-secondary leading-relaxed">
                        {severityWhy}
                      </p>
                      <p className="text-[11px] text-gray-400 italic">
                        Severity determined by backend evidence-fusion logic.
                      </p>
                    </div>

                    {/* Filter Decision Explanation */}
                    <div className="rounded-lg bg-card border border-border p-3 space-y-1.5 shadow-2xs">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-text-navy flex items-center gap-1.5">
                          <CheckCircle className="h-3.5 w-3.5 text-emerald-600" />
                          Filter Status:
                        </span>
                        <span className="font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded text-[11px]">
                          {d.status}
                        </span>
                      </div>
                      <p className="text-text-secondary leading-relaxed">
                        {d.status === "RETAINED"
                          ? "Detection passed the backend false-positive filtering stage and was kept in the final detection results."
                          : "Detection was categorized as natural seafloor background by the filter."}
                      </p>
                      <div className="text-[11px] text-text-secondary">
                        Parameters checked: area ({area} px²), aspect ratio ({aspectRatio}), shadow index ({(d.shadow_score * 100).toFixed(0)}%).
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
