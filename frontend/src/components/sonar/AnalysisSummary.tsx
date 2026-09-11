import type { AnalysisResponse } from "../../api/types";

interface AnalysisSummaryProps {
  result: AnalysisResponse;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-text-secondary">{label}</span>
      <span className="font-medium text-text-navy">{value}</span>
    </div>
  );
}

export function AnalysisSummary({ result }: AnalysisSummaryProps) {
  const avgConfidence =
    result.detections.length > 0
      ? result.detections.reduce((sum, d) => sum + d.confidence_pct, 0) /
        result.detections.length
      : null;

  const quality = result.quality as { quality?: string; score?: number } | undefined;
  const shadowScores = result.detections
    .map((d) => d.shadow_score)
    .filter((s) => s !== undefined && s !== null);
  const avgShadow =
    shadowScores.length > 0
      ? shadowScores.reduce((a, b) => a + b, 0) / shadowScores.length
      : null;

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h2 className="mb-2 text-sm font-semibold text-text-navy">Analysis Summary</h2>
      <div className="divide-y divide-border">
        <Row label="Objects Detected" value={String(result.detections.length)} />
        <Row
          label="Average Confidence"
          value={avgConfidence !== null ? `${avgConfidence.toFixed(1)}%` : "Not available"}
        />
        <Row
          label="Image Quality"
          value={quality?.quality ?? "Not available"}
        />
        <Row
          label="Shadow Evidence"
          value={avgShadow !== null ? `${(avgShadow * 100).toFixed(0)}%` : "Not available"}
        />
        <Row
          label="Anomalies"
          value={String(result.num_anomalies)}
        />
        <Row
          label="Processing Time"
          value={`${result.processing_time_ms.toFixed(0)} ms`}
        />
        <Row label="Model Mode" value={result.mode} />
      </div>
      {result.warnings.length > 0 && (
        <div className="mt-3 rounded-md bg-status-warning/10 p-2 text-xs text-status-warning">
          {result.warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
