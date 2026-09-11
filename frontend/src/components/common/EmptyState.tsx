import type { LucideIcon } from "lucide-react";
import { Inbox } from "lucide-react";

interface EmptyStateProps {
  message: string;
  icon?: LucideIcon;
}

export function EmptyState({ message, icon: Icon = Inbox }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-bg-secondary/40 px-6 py-10 text-center">
      <Icon className="h-6 w-6 text-text-secondary" strokeWidth={1.5} />
      <p className="text-sm text-text-secondary">{message}</p>
    </div>
  );
}
