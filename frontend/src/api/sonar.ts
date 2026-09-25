import { apiGet, apiPostForm, getApiBaseUrl } from "./client";
import type { AnalysisResponse } from "./types";

export type ImageVariant = "raw" | "processed" | "annotated";

export function analysisImageUrl(
  analysisId: string,
  variant: ImageVariant = "raw"
): string {
  const base = getApiBaseUrl();
  return `${base}/analysis/${encodeURIComponent(analysisId)}/image?variant=${variant}`;
}

export interface AnalyzeSonarParams {
  file: File;
  missionId?: string;
  lat?: number;
  lon?: number;
  depthM?: number;
}

export async function analyzeSonar(
  params: AnalyzeSonarParams
): Promise<AnalysisResponse> {
  const form = new FormData();
  form.append("file", params.file);
  if (params.missionId) form.append("mission_id", params.missionId);
  if (params.lat !== undefined) form.append("lat", String(params.lat));
  if (params.lon !== undefined) form.append("lon", String(params.lon));
  if (params.depthM !== undefined) form.append("depth_m", String(params.depthM));

  return apiPostForm<AnalysisResponse>("/sonar/analyze", form);
}

export async function getLatestAnalysis(): Promise<AnalysisResponse | null> {
  return apiGet<AnalysisResponse | null>("/analysis/latest");
}
