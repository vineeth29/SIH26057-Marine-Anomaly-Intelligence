import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Waves,
  ListTree,
  Search,
  Map as MapIcon,
  FileText,
  Cpu,
  Activity,
} from "lucide-react";
import { cn } from "../../utils/cn";

interface NavItem {
  label: string;
  to: string;
  icon: typeof LayoutDashboard;
}

const OPERATIONS: NavItem[] = [
  { label: "Dashboard", to: "/", icon: LayoutDashboard },
  { label: "Sonar Analysis", to: "/sonar-analysis", icon: Waves },
  { label: "Detections", to: "/detections", icon: ListTree },
  { label: "Anomaly Analysis", to: "/anomaly-analysis", icon: Search },
  { label: "Map", to: "/map", icon: MapIcon },
];

const OUTPUT: NavItem[] = [{ label: "Reports", to: "/reports", icon: FileText }];

const SYSTEM: NavItem[] = [
  { label: "Model", to: "/model", icon: Cpu },
  { label: "System Status", to: "/system-status", icon: Activity },
];

function NavSection({
  title,
  items,
  onNavigate,
}: {
  title: string;
  items: NavItem[];
  onNavigate?: () => void;
}) {
  return (
    <div className="mb-5">
      <div className="mb-1.5 px-3 text-[10px] font-semibold uppercase tracking-wider text-text-secondary">
        {title}
      </div>
      <nav className="flex flex-col gap-0.5">
        {items.map(({ label, to, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                "relative flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-ocean/10 text-ocean font-semibold"
                  : "text-text-secondary hover:bg-bg-secondary hover:text-text-navy"
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <div className="absolute left-0 top-1.5 bottom-1.5 w-[3px] rounded-full bg-ocean transition-all duration-200 ease-out" />
                )}
                <Icon className="h-4 w-4" strokeWidth={1.75} />
                {label}
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

interface SidebarProps {
  mobileOpen?: boolean;
  onClose?: () => void;
}

export function Sidebar({ mobileOpen = false, onClose }: SidebarProps) {
  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/40 backdrop-blur-[2px] transition-opacity duration-200 md:hidden"
          aria-hidden="true"
        />
      )}
      <aside
        className={cn(
          "flex w-60 shrink-0 flex-col border-r border-border bg-card px-3 py-5 transition-transform duration-200 ease-in-out",
          "fixed inset-y-16 left-0 z-50 md:static md:translate-x-0",
          mobileOpen ? "translate-x-0 shadow-2xl" : "-translate-x-full md:translate-x-0"
        )}
      >
        <NavSection title="Operations" items={OPERATIONS} onNavigate={onClose} />
        <NavSection title="Output" items={OUTPUT} onNavigate={onClose} />
        <NavSection title="System" items={SYSTEM} onNavigate={onClose} />
      </aside>
    </>
  );
}
