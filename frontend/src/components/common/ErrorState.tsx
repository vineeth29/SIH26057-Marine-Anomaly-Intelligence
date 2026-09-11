import { AlertTriangle, WifiOff } from "lucide-react";
import { ApiError } from "../../api/client";

interface ErrorStateProps {
  error: unknown;
  onRetry?: () => void;
}

function messageFor(error: unknown): { title: string; detail: string; offline: boolean } {
  if (error instanceof ApiError) {
    if (error.status === 0) {
      return {
        title: "Unable to connect to the analysis service.",
        detail: "Check that the backend API is running and reachable.",
        offline: true,
      };
    }
    return { title: "Request failed", detail: error.detail, offline: false };
  }
  if (error instanceof TypeError) {
    // fetch() network failure
    return {
      title: "Unable to connect to the analysis service.",
      detail: "Check that the backend API is running and reachable.",
      offline: true,
    };
  }
  return {
    title: "Something went wrong.",
    detail: error instanceof Error ? error.message : String(error),
    offline: false,
  };
}

export function ErrorState({ error, onRetry }: ErrorStateProps) {
  const { title, detail, offline } = messageFor(error);
  const Icon = offline ? WifiOff : AlertTriangle;

  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-status-high/30 bg-status-high/5 px-6 py-10 text-center">
      <Icon className="h-6 w-6 text-status-high" strokeWidth={1.75} />
      <p className="text-sm font-medium text-text-navy">{title}</p>
      <p className="text-xs text-text-secondary">{detail}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy hover:bg-bg-secondary"
        >
          Retry
        </button>
      )}
    </div>
  );
}
