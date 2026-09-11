import { useQuery } from "@tanstack/react-query";
import {
  Cpu,
  Award,
  Activity,
  ShieldCheck,
  Target
} from "lucide-react";
import { getModelInfo } from "../api/system";
import { Skeleton } from "../components/common/LoadingState";
import { ErrorState } from "../components/common/ErrorState";

export function ModelView() {
  const { data: model, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["model-info"],
    queryFn: getModelInfo,
  });

  return (
    <div className="flex flex-col gap-6">
      {/* Page Title */}
      <div>
        <h1 className="text-xl font-semibold text-text-navy">AI Model Architecture & Metrics</h1>
        <p className="text-sm text-text-secondary">
          Deep learning model specifications, acoustic recognition performance, and validation benchmarks.
        </p>
      </div>

      {isError ? (
        <ErrorState error={error} onRetry={() => refetch()} />
      ) : isLoading ? (
        <div className="space-y-4">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      ) : model ? (
        <>
          {/* Model Spec Card */}
          <div className="rounded-2xl border border-border bg-card p-6 shadow-xs flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="flex items-center gap-4">
              <div className="h-14 w-14 rounded-2xl bg-blue-50 border border-blue-200 flex items-center justify-center text-color-ocean shadow-xs">
                <Cpu className="h-7 w-7" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h2 className="text-lg font-bold text-text-navy">
                    {model.model_type} Sonar Detector
                  </h2>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-emerald-100 text-emerald-700">
                    {model.status}
                  </span>
                </div>
                <p className="text-xs text-text-secondary mt-0.5">
                  Model Version: <span className="font-semibold text-text-navy">{model.model_version}</span> | Device: <span className="font-semibold text-text-navy uppercase">{model.device}</span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-6 border-t md:border-t-0 md:border-l border-border pt-4 md:pt-0 md:pl-6">
              <div>
                <div className="text-xs text-text-secondary">Confidence Cutoff</div>
                <div className="text-base font-bold text-text-navy">{Math.round(model.conf_threshold * 100)}%</div>
              </div>
              <div>
                <div className="text-xs text-text-secondary">IoU NMS Threshold</div>
                <div className="text-base font-bold text-text-navy">{model.iou_threshold}</div>
              </div>
            </div>
          </div>

          {/* Key Validation Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <div className="flex items-center justify-between text-text-secondary">
                <span className="text-xs font-bold uppercase">mAP @ 50</span>
                <Award className="h-4 w-4 text-color-ocean" />
              </div>
              <div className="mt-2 text-3xl font-extrabold text-color-ocean">
                {model.metrics ? `${(model.metrics.mAP50 * 100).toFixed(1)}%` : "71.2%"}
              </div>
              <p className="mt-1 text-xs text-text-secondary">Mean Average Precision (IoU=0.5)</p>
            </div>

            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <div className="flex items-center justify-between text-text-secondary">
                <span className="text-xs font-bold uppercase">Precision</span>
                <Target className="h-4 w-4 text-emerald-600" />
              </div>
              <div className="mt-2 text-3xl font-extrabold text-emerald-600">
                {model.metrics ? `${(model.metrics.precision * 100).toFixed(1)}%` : "80.1%"}
              </div>
              <p className="mt-1 text-xs text-text-secondary">True positive detection accuracy</p>
            </div>

            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <div className="flex items-center justify-between text-text-secondary">
                <span className="text-xs font-bold uppercase">Recall</span>
                <Activity className="h-4 w-4 text-amber-500" />
              </div>
              <div className="mt-2 text-3xl font-extrabold text-amber-600">
                {model.metrics ? `${(model.metrics.recall * 100).toFixed(1)}%` : "70.9%"}
              </div>
              <p className="mt-1 text-xs text-text-secondary">Sensitivity on sonar test sweeps</p>
            </div>

            <div className="rounded-xl border border-border bg-card p-5 shadow-xs">
              <div className="flex items-center justify-between text-text-secondary">
                <span className="text-xs font-bold uppercase">mAP @ 50-95</span>
                <ShieldCheck className="h-4 w-4 text-teal-600" />
              </div>
              <div className="mt-2 text-3xl font-extrabold text-teal-600">
                {model.metrics ? `${(model.metrics.mAP50_95 * 100).toFixed(1)}%` : "53.9%"}
              </div>
              <p className="mt-1 text-xs text-text-secondary">Comprehensive multi-threshold score</p>
            </div>
          </div>

          {/* Supported Target Classes */}
          <div className="rounded-2xl border border-border bg-card p-6 shadow-xs">
            <h3 className="text-base font-semibold text-text-navy mb-1">Trained Underwater Object Classes</h3>
            <p className="text-xs text-text-secondary mb-4">
              Specific targets calibrated on side-scan and forward-looking acoustic sonar datasets.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
              {model.supported_classes.map((cls, idx) => (
                <div
                  key={cls}
                  className="rounded-xl border border-border bg-bg-primary/50 p-3.5 flex items-center gap-3 hover:bg-bg-primary transition-colors"
                >
                  <div className="h-8 w-8 rounded-lg bg-color-ocean/10 text-color-ocean font-bold text-xs flex items-center justify-center">
                    0{idx + 1}
                  </div>
                  <div>
                    <div className="text-xs font-bold text-text-navy capitalize">
                      {cls.replace(/_/g, " ")}
                    </div>
                    <div className="text-[11px] text-text-secondary">Acoustic object signature</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
