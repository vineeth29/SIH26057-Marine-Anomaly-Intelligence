import { useQuery } from "@tanstack/react-query";
import { Menu, X } from "lucide-react";
import { getSystemStatus } from "../../api/system";
import { StatusBadge } from "../common/StatusBadge";

interface HeaderProps {
  sidebarOpen?: boolean;
  onToggleSidebar?: () => void;
}

export function Header({ sidebarOpen = false, onToggleSidebar }: HeaderProps) {
  const { data, isLoading } = useQuery({
    queryKey: ["system-status-header"],
    queryFn: getSystemStatus,
    refetchInterval: 30_000,
    retry: 1,
  });

  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border bg-card px-4 md:px-6 z-20">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSidebar}
          aria-label="Toggle navigation sidebar"
          className="flex h-9 w-9 items-center justify-center rounded-lg bg-ocean/10 hover:bg-ocean/20 transition-colors focus:outline-none cursor-pointer border border-ocean/20 text-ocean shadow-2xs"
        >
          {sidebarOpen ? (
            <X className="h-5 w-5" strokeWidth={2.2} />
          ) : (
            <Menu className="h-5 w-5" strokeWidth={2.2} />
          )}
        </button>
        <div>
          <div className="text-sm font-semibold leading-tight text-text-navy tracking-tight">
            Marine Anomaly Intelligence
          </div>
          <div className="text-xs leading-tight text-text-secondary">
            AI-Powered Underwater Sonar Analysis | SIH26057
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {isLoading ? (
          <span className="text-xs text-text-secondary">Checking status...</span>
        ) : data ? (
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-secondary hidden sm:inline">System</span>
            <StatusBadge kind="component" value={data.overall} />
          </div>
        ) : (
          <StatusBadge kind="component" value="UNAVAILABLE" />
        )}
      </div>
    </header>
  );
}
