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

function NavSection({ title, items }: { title: string; items: NavItem[] }) {
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
            className={({ isActive }) =>
              cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-ocean/10 text-ocean"
                  : "text-text-secondary hover:bg-bg-secondary hover:text-text-navy"
              )
            }
          >
            <Icon className="h-4 w-4" strokeWidth={1.75} />
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}

export function Sidebar() {
  return (
    <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-card px-3 py-5">
      <NavSection title="Operations" items={OPERATIONS} />
      <NavSection title="Output" items={OUTPUT} />
      <NavSection title="System" items={SYSTEM} />
    </aside>
  );
}
