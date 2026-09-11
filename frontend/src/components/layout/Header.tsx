import { useQuery } from "@tanstack/react-query";
import { Radio } from "lucide-react";
import { getSystemStatus } from "../../api/system";
import { StatusBadge } from "../common/StatusBadge";

export function Header() {
  const { data, isLoading } = useQuery({
    queryKey: ["system-status-header"],
    queryFn: getSystemStatus,
    refetchInterval: 30_000,
    retry: 1,
  });

  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-card px-6">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-md bg-ocean/10">
          <Radio className="h-4 w-4 text-ocean" strokeWidth={2} />
        </div>
        <div>
          <div className="text-sm font-semibold leading-tight text-text-navy">
            Marine Anomaly Intelligence
          </div>
          <div className="text-xs leading-tight text-text-secondary">
            AI-Powered Underwater Sonar Analysis | SIH26057
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {isLoading ? (
          <span className="text-xs text-text-secondary">Checking status…</span>
        ) : data ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary">System</span>
            <StatusBadge kind="component" value={data.overall} />
          </div>
        ) : (
          <StatusBadge kind="component" value="UNAVAILABLE" />
        )}
      </div>
    </header>
  );
}
