import { Loader2 } from "lucide-react";

export function LoadingState({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-border bg-card px-6 py-10 text-center">
      <Loader2 className="h-5 w-5 animate-spin text-ocean" strokeWidth={2} />
      <p className="text-sm text-text-secondary">{label}</p>
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div className={`animate-pulse rounded-md bg-bg-secondary ${className}`} />
  );
}
