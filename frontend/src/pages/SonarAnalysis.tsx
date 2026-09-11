import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { analyzeSonar, analysisImageUrl } from "../api/sonar";
import type { AnalysisResponse } from "../api/types";
import { UploadPanel } from "../components/sonar/UploadPanel";
import { AnalysisProgress } from "../components/sonar/AnalysisProgress";
import { SonarViewer } from "../components/sonar/SonarViewer";
import { AnalysisSummary } from "../components/sonar/AnalysisSummary";
import { DetectionList } from "../components/detections/DetectionList";
import { ErrorState } from "../components/common/ErrorState";

type ImageTab = "raw" | "processed" | "annotated";

export function SonarAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<ImageTab>("raw");

  const mutation = useMutation<AnalysisResponse, unknown, File>({
    mutationFn: (f) => analyzeSonar({ file: f }),
  });

  const result = mutation.data;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold text-text-navy">Sonar Analysis</h1>
        <p className="text-sm text-text-secondary">
          Upload a side-scan sonar image to run detection, evidence, and
          geolocation analysis.
        </p>
      </div>

      {!result && !mutation.isPending && (
        <div className="flex flex-col gap-4">
          <UploadPanel onFileSelected={setFile} selectedFile={file} />
          <div>
            <button
              disabled={!file}
              onClick={() => file && mutation.mutate(file)}
              className="rounded-md bg-action px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-action/90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Analyze Sonar
            </button>
          </div>
        </div>
      )}

      {mutation.isPending && <AnalysisProgress />}

      {mutation.isError && (
        <ErrorState
          error={mutation.error}
          onRetry={() => file && mutation.mutate(file)}
        />
      )}

      {result && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex gap-1 rounded-md border border-border bg-card p-1">
              {(["raw", "processed", "annotated"] as ImageTab[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`rounded px-3 py-1 text-xs font-medium capitalize transition-colors ${
                    activeTab === tab
                      ? "bg-ocean/10 text-ocean"
                      : "text-text-secondary hover:text-text-navy"
                  }`}
                >
                  {tab === "raw" ? "Original" : tab === "processed" ? "Processed" : "Detection"}
                </button>
              ))}
            </div>
            <button
              onClick={() => {
                mutation.reset();
                setFile(null);
              }}
              className="rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy hover:bg-bg-secondary"
            >
              New Analysis
            </button>
          </div>

          <div className="grid grid-cols-3 gap-6">
            <div className="col-span-2">
              <SonarViewer
                imageUrl={analysisImageUrl(
                  result.analysis_id,
                  activeTab === "annotated" ? "annotated" : activeTab
                )}
                detections={activeTab === "raw" ? result.detections : []}
                imageWidth={result.image.width}
                imageHeight={result.image.height}
              />
            </div>
            <div className="flex flex-col gap-4">
              <AnalysisSummary result={result} />
            </div>
          </div>

          <div className="rounded-lg border border-border bg-card p-4">
            <h2 className="mb-2 text-sm font-semibold text-text-navy">
              Detection Results
            </h2>
            <DetectionList detections={result.detections} />
          </div>
        </div>
      )}
    </div>
  );
}
