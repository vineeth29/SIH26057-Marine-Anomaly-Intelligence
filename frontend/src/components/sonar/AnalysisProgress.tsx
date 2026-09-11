import { Loader2 } from "lucide-react";

// These stages describe what /api/sonar/analyze does server-side
// (see services/pipeline_service.py SonarAnalysisPipeline.run), shown
// for context — NOT as a live per-stage tracker, since the backend is
// a single synchronous call and emits no per-stage events. Marking any
// one of these as "done" before the request actually completes would
// be a fabricated signal, so all stages render identically and only
// the overall spinner reflects real state (request in flight).
const STAGES = [
  "Data ingestion",
  "Image quality assessment",
  "Sonar preprocessing",
  "AI object detection",
  "Acoustic shadow analysis",
  "Evidence fusion",
  "Geolocation",
  "Final result",
];

export function AnalysisProgress() {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="mb-3 flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin text-ocean" strokeWidth={2} />
        <span className="text-sm font-semibold text-text-navy">
          Analyzing Sonar Data
        </span>
      </div>
      <ul className="flex flex-col gap-1.5">
        {STAGES.map((stage) => (
          <li key={stage} className="flex items-center gap-2 text-xs text-text-secondary">
            <span className="h-1.5 w-1.5 rounded-full bg-border" />
            {stage}
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11px] text-text-secondary">
        Processing time depends on image size and server load. Stages above
        describe the analysis pipeline; per-stage progress is not currently
        reported by the backend.
      </p>
    </div>
  );
}
