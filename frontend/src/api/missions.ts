import { apiGet, apiPostJson, getApiBaseUrl } from "./client";
import type {
  MissionSummary,
  MissionDetail,
  DetectionRecord,
  OperatorReviewRequest,
} from "./types";

export function getMissions(): Promise<MissionSummary[]> {
  return apiGet<MissionSummary[]>("/missions");
}

export function getMissionDetail(missionId: string): Promise<MissionDetail> {
  return apiGet<MissionDetail>(`/missions/${encodeURIComponent(missionId)}`);
}

export function getMissionDetections(missionId: string): Promise<DetectionRecord[]> {
  return apiGet<DetectionRecord[]>(`/missions/${encodeURIComponent(missionId)}/detections`);
}

export function getAllDetections(
  missionId?: string,
  limit: number = 200
): Promise<DetectionRecord[]> {
  const query = new URLSearchParams();
  if (missionId) query.append("mission_id", missionId);
  query.append("limit", String(limit));
  return apiGet<DetectionRecord[]>(`/detections?${query.toString()}`);
}

export function reviewDetection(
  detectionId: string,
  req: OperatorReviewRequest
): Promise<{ status: string; detection_id: string }> {
  return apiPostJson<{ status: string; detection_id: string }>(
    `/detections/${encodeURIComponent(detectionId)}/review`,
    req
  );
}

export function getCsvExportUrl(missionId?: string): string {
  const base = getApiBaseUrl();
  return missionId
    ? `${base}/missions/${encodeURIComponent(missionId)}/export/csv`
    : `${base}/export/csv`;
}

export function getJsonExportUrl(missionId?: string): string {
  const base = getApiBaseUrl();
  return missionId
    ? `${base}/missions/${encodeURIComponent(missionId)}/export/json`
    : `${base}/export/json`;
}

export function getPdfReportUrl(analysisId: string, missionId: string = "MISSION-001"): string {
  const base = getApiBaseUrl();
  return `${base}/reports/analysis/${encodeURIComponent(analysisId)}/pdf?mission_id=${encodeURIComponent(missionId)}`;
}
