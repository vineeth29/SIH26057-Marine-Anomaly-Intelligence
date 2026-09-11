import type { Detection } from "../../api/types";
import { StatusBadge } from "../common/StatusBadge";
import { EmptyState } from "../common/EmptyState";

interface DetectionListProps {
  detections: Detection[];
}

export function DetectionList({ detections }: DetectionListProps) {
  if (detections.length === 0) {
    return <EmptyState message="No anomalies were detected in this sonar image." />;
  }

  return (
    <div className="flex flex-col divide-y divide-border">
      {detections.map((d) => (
        <div key={d.detection_id} className="flex items-center justify-between py-2.5">
          <div>
            <div className="text-sm font-medium text-text-navy">{d.display_name}</div>
            <div className="text-xs text-text-secondary">
              {d.confidence_pct.toFixed(0)}% confidence
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusBadge kind="severity" value={d.severity} />
            <StatusBadge kind="detection" value={d.status} />
          </div>
        </div>
      ))}
    </div>
  );
}
