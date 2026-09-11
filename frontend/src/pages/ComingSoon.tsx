import { Construction } from "lucide-react";

export function ComingSoon({ pageName }: { pageName: string }) {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-text-navy">{pageName}</h1>
      </div>
      <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-bg-secondary/40 px-6 py-16 text-center">
        <Construction className="h-6 w-6 text-text-secondary" strokeWidth={1.5} />
        <p className="text-sm text-text-secondary">
          {pageName} is not yet implemented in this build.
        </p>
      </div>
    </div>
  );
}
