import { apiPostForm } from "./client";
import type { AnalysisResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL as string;

export type ImageVariant = "raw" | "processed" | "annotated";

/**
 * Direct URL to a result image variant for a completed analysis.
 * Not run through apiGet since it's rendered via <img src=...>, not
 * parsed as JSON. Default "raw" — the interactive viewer draws its own
 * bbox overlay on top of this using AnalysisResponse.detections[].bbox.
 */
export function analysisImageUrl(
  analysisId: string,
  variant: ImageVariant = "raw"
): string {
  return `${API_BASE_URL}/analysis/${analysisId}/image?variant=${variant}`;
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
