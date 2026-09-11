import { cn } from "../../utils/cn";
import type { Severity, ComponentStatusValue, DetectionStatus } from "../../api/types";

const SEVERITY_STYLES: Record<Severity, string> = {
  HIGH: "bg-status-high/10 text-status-high border-status-high/30",
  MEDIUM: "bg-status-warning/10 text-status-warning border-status-warning/30",
  LOW: "bg-status-success/10 text-status-success border-status-success/30",
  UNKNOWN: "bg-text-secondary/10 text-text-secondary border-text-secondary/30",
};

const COMPONENT_STATUS_STYLES: Record<ComponentStatusValue, string> = {
  READY: "bg-status-success/10 text-status-success border-status-success/30",
  WARNING: "bg-status-warning/10 text-status-warning border-status-warning/30 animate-[subtleWarningPulse_3s_ease-in-out_infinite]",
  ERROR: "bg-status-high/10 text-status-high border-status-high/30 animate-[subtleWarningPulse_2.5s_ease-in-out_infinite]",
  UNAVAILABLE: "bg-text-secondary/10 text-text-secondary border-text-secondary/30",
};

const COMPONENT_STATUS_LABELS: Record<string, string> = {
  READY: "OPERATIONAL",
  OPERATIONAL: "OPERATIONAL",
  WARNING: "WARNING",
  ERROR: "ERROR",
  UNAVAILABLE: "OFFLINE",
};

const DETECTION_STATUS_STYLES: Record<DetectionStatus, string> = {
  RETAINED: "bg-ocean/10 text-ocean border-ocean/30",
  FILTERED: "bg-text-secondary/10 text-text-secondary border-text-secondary/30",
};

interface StatusBadgeProps {
  kind: "severity" | "component" | "detection";
  value: Severity | ComponentStatusValue | DetectionStatus;
  className?: string;
}

export function StatusBadge({ kind, value, className }: StatusBadgeProps) {
  const styles =
    kind === "severity"
      ? SEVERITY_STYLES[value as Severity]
      : kind === "component"
      ? COMPONENT_STATUS_STYLES[value as ComponentStatusValue]
      : DETECTION_STATUS_STYLES[value as DetectionStatus];

  const displayValue =
    kind === "component"
      ? COMPONENT_STATUS_LABELS[String(value)] || value
      : value;

  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold tracking-wide transition-colors",
        styles,
        className
      )}
    >
      {displayValue}
    </span>
  );
}
