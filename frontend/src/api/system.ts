import { apiGet } from "./client";
import type {
  HealthResponse,
  ModelInfo,
  SystemStatusResponse,
  DashboardStats,
} from "./types";

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/health");
}

export function getModelInfo(): Promise<ModelInfo> {
  return apiGet<ModelInfo>("/model");
}

export function getSystemStatus(): Promise<SystemStatusResponse> {
  return apiGet<SystemStatusResponse>("/system/status");
}

export function getDashboardStats(): Promise<DashboardStats> {
  return apiGet<DashboardStats>("/dashboard/stats");
}
