import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import type { Map as LeafletMap } from "leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import {
  Layers,
  Crosshair,
  XCircle,
  AlertCircle,
  Compass,
} from "lucide-react";
import { getMissions, getAllDetections, reviewDetection } from "../api/missions";
import type { DetectionRecord } from "../api/types";
import { Skeleton } from "../components/common/LoadingState";
import { ErrorState } from "../components/common/ErrorState";
import { StatusBadge } from "../components/common/StatusBadge";

// ─── Custom Marker Icons ──────────────────────────────────────────────────────
const createCustomIcon = (
  color: string,
  isSelected: boolean,
  isSimulated: boolean,
  index: number
) => {
  const staggerDelay = Math.min(index * 40, 320);
  const size = isSelected ? 20 : 15;
  const borderStyle = isSimulated ? "2px dashed white" : "2px solid white";
  const glow = isSelected ? `0 0 12px ${color}` : "0 0 6px rgba(0,0,0,0.45)";
  return L.divIcon({
    className: "custom-map-marker",
    html: `<div style="background-color:${color};width:${size}px;height:${size}px;border-radius:50%;border:${borderStyle};box-shadow:${glow};animation:markerEntrance 200ms ease-out both;animation-delay:${staggerDelay}ms;"></div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -(size / 2 + 4)],
  });
};

const CLASS_LABELS: Record<string, string> = {
  crab_pot: "Crab Pot",
  submarine_pipeline: "Submarine Pipeline",
  shipwreck: "Shipwreck",
  ghost_net: "Ghost Net",
  mine_cylinder: "Mine Cylinder",
};

// ─── Simulated Offshore Demo Targets ─────────────────────────────────────────
const SIMULATED_OFFSHORE_TARGETS: DetectionRecord[] = [
  {
    detection_id: "DEMO-001", mission_id: "DEMO-SURVEY",
    class_name: "submarine_pipeline", confidence: 0.89,
    is_anomaly: false, anomaly_score: 0.04, shadow_score: 0.85,
    evidence_score: 0.84, severity: "HIGH", bbox: [210, 140, 480, 290],
    lat: 13.1520, lon: 80.4850, depth_m: 42.5,
    location_source: "SIMULATED", coordinates_label: "SIMULATED",
    mode: "SIMULATED", operator_status: "pending",
  },
  {
    detection_id: "DEMO-002", mission_id: "DEMO-SURVEY",
    class_name: "shipwreck", confidence: 0.93,
    is_anomaly: false, anomaly_score: 0.08, shadow_score: 0.92,
    evidence_score: 0.91, severity: "HIGH", bbox: [310, 180, 560, 310],
    lat: 13.1490, lon: 80.4920, depth_m: 55.0,
    location_source: "SIMULATED", coordinates_label: "SIMULATED",
    mode: "SIMULATED", operator_status: "pending",
  },
  {
    detection_id: "DEMO-003", mission_id: "DEMO-SURVEY",
    class_name: "mine_cylinder", confidence: 0.78,
    is_anomaly: true, anomaly_score: 0.72, shadow_score: 0.68,
    evidence_score: 0.74, severity: "HIGH", bbox: [120, 90, 200, 165],
    lat: 13.1560, lon: 80.4780, depth_m: 38.0,
    location_source: "SIMULATED", coordinates_label: "SIMULATED",
    mode: "SIMULATED", operator_status: "investigate",
  },
  {
    detection_id: "DEMO-004", mission_id: "DEMO-SURVEY",
    class_name: "ghost_net", confidence: 0.67,
    is_anomaly: false, anomaly_score: 0.21, shadow_score: 0.55,
    evidence_score: 0.60, severity: "MEDIUM", bbox: [420, 220, 600, 300],
    lat: 13.1610, lon: 80.4960, depth_m: 29.5,
    location_source: "SIMULATED", coordinates_label: "SIMULATED",
    mode: "SIMULATED", operator_status: "pending",
  },
  {
    detection_id: "DEMO-005", mission_id: "DEMO-SURVEY",
    class_name: "crab_pot", confidence: 0.86,
    is_anomaly: false, anomaly_score: 0.06, shadow_score: 0.78,
    evidence_score: 0.81, severity: "MEDIUM", bbox: [100, 160, 540, 240],
    lat: 13.1580, lon: 80.5050, depth_m: 48.0,
    location_source: "SIMULATED", coordinates_label: "SIMULATED",
    mode: "SIMULATED", operator_status: "confirmed",
  },
];

// ─── MapPopupCloser ───────────────────────────────────────────────────────────
// Lives INSIDE <MapContainer> so it can call useMap().
// Passes the Leaflet map instance up to the parent via onMapReady callback.
function MapPopupCloser({ onMapReady }: { onMapReady: (map: LeafletMap) => void }) {
  const map = useMap();
  useEffect(() => { onMapReady(map); }, [map, onMapReady]);
  return null;
}

// ─── MapRecenter ──────────────────────────────────────────────────────────────
function MapRecenter({ center }: { center: [number, number] }) {
  const map = useMap();
  useMemo(() => { map.flyTo(center, 12, { duration: 1.0 }); }, [center, map]);
  return null;
}

// ─── DetectionReviewModal (Portal) ───────────────────────────────────────────
// Rendered via createPortal to document.body.
// z-index 9999 (backdrop) / 10000 (dialog) — above all Leaflet layers (max ~800).
interface ReviewModalProps {
  detection: DetectionRecord;
  reviewStatus: string;
  reviewNote: string;
  isSaving: boolean;
  onStatusChange: (s: string) => void;
  onNoteChange: (n: string) => void;
  onSave: () => void;
  onClose: () => void;
}

function DetectionReviewModal({
  detection, reviewStatus, reviewNote, isSaving,
  onStatusChange, onNoteChange, onSave, onClose,
}: ReviewModalProps) {
  const isSimulated = (detection.location_source || detection.coordinates_label) === "SIMULATED";

  // Escape key closes modal
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Prevent body scroll while modal is open
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = ""; };
  }, []);

  const modal = (
    // Backdrop — z-index 9999, captures all pointer events, blocks map
    <div
      style={{ position: "fixed", inset: 0, zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center", padding: "1rem", backgroundColor: "rgba(22,50,79,0.45)" }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
      aria-modal="true"
      role="dialog"
      aria-label="Detection Review"
    >
      {/* Dialog — z-index 10000 */}
      <div
        style={{ position: "relative", zIndex: 10000, width: "100%", maxWidth: "32rem", borderRadius: "0.75rem", border: "1px solid var(--color-border)", backgroundColor: "var(--color-card)", padding: "1.5rem", boxShadow: "0 20px 40px rgba(0,0,0,0.18)", maxHeight: "90vh", overflowY: "auto" }}
        className="animate-[pageEntrance_150ms_ease-out]"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-border pb-3">
          <div>
            <h3 className="text-base font-semibold text-text-navy">
              Detection Review &amp; Explainability
            </h3>
            <span className="text-xs font-mono text-text-secondary">
              {detection.detection_id} &middot; {detection.mission_id || "MISSION-001"}
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md p-1 text-text-secondary hover:bg-bg-secondary transition-colors"
            aria-label="Close review modal"
          >
            <XCircle className="h-5 w-5" />
          </button>
        </div>

        {/* Simulated badge */}
        {isSimulated && (
          <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 font-medium">
            ⚠ SIMULATED target — Demo coordinate, not field GPS data
          </div>
        )}

        {/* Explainability Grid */}
        <div className="my-4 grid grid-cols-2 gap-3 text-xs">
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">Classification</span>
            <span className="text-sm font-semibold text-text-navy">
              {CLASS_LABELS[detection.class_name] || detection.class_name.replace(/_/g, " ")}
            </span>
          </div>
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">Assigned Severity</span>
            <div className="mt-1"><StatusBadge kind="severity" value={detection.severity} /></div>
          </div>
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">YOLO Confidence</span>
            <span className="text-sm font-mono font-bold text-ocean">
              {Math.round(detection.confidence * 100)}%
            </span>
          </div>
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">Acoustic Evidence Score</span>
            <span className="text-sm font-mono font-bold text-text-navy">
              {detection.evidence_score != null
                ? `${Math.round(detection.evidence_score * 100)}%` : "N/A"}
            </span>
          </div>
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">Acoustic Shadow Score</span>
            <span className="text-sm font-mono font-bold text-text-navy">
              {detection.shadow_score != null
                ? `${Math.round((detection.shadow_score as number) * 100)}%` : "N/A"}
            </span>
          </div>
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">Anomaly Score</span>
            <span className="text-sm font-mono font-bold text-text-navy">
              {detection.anomaly_score != null
                ? `${Math.round((detection.anomaly_score as number) * 100)}%` : "N/A"}
            </span>
          </div>
          <div className="rounded-lg border border-border bg-bg-primary p-3">
            <span className="text-text-secondary font-medium block">Location Source</span>
            <span className={`text-sm font-mono font-bold ${isSimulated ? "text-amber-700" : "text-ocean"}`}>
              {detection.location_source || "GPS"}
            </span>
          </div>
          {detection.depth_m != null && (
            <div className="rounded-lg border border-border bg-bg-primary p-3">
              <span className="text-text-secondary font-medium block">Depth</span>
              <span className="text-sm font-mono font-bold text-text-navy">{detection.depth_m}m</span>
            </div>
          )}
        </div>

        {/* Operator Feedback */}
        <div className="space-y-3 border-t border-border pt-3">
          <label className="block text-xs font-semibold text-text-navy">
            Operator Validation Status:
          </label>
          <div className="grid grid-cols-4 gap-2">
            {[
              { value: "confirmed", label: "Confirmed" },
              { value: "pending", label: "Pending" },
              { value: "investigate", label: "Investigate" },
              { value: "false_positive", label: "False Alarm" },
            ].map((s) => (
              <button
                key={s.value}
                type="button"
                onClick={() => onStatusChange(s.value)}
                className={`flex-1 rounded-lg border py-1.5 text-xs font-semibold transition-colors ${
                  reviewStatus === s.value
                    ? "border-ocean bg-ocean/10 text-ocean shadow-2xs"
                    : "border-border bg-bg-primary text-text-secondary hover:bg-bg-secondary"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
          <label className="block text-xs font-semibold text-text-navy mb-1.5">
            Operator Mission Notes:
          </label>
          <textarea
            rows={3}
            value={reviewNote}
            onChange={(e) => onNoteChange(e.target.value)}
            placeholder="Enter verification notes, seabed context, or action instructions..."
            className="w-full rounded-lg border border-border bg-bg-primary p-2.5 text-xs text-text-navy focus:outline-none focus:ring-1 focus:ring-ocean"
          />
        </div>

        {/* Action Buttons */}
        <div className="mt-4 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:bg-bg-secondary transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={isSaving}
            className="rounded-lg bg-action px-4 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-action/90 disabled:opacity-50 transition-colors"
          >
            {isSaving ? "Saving..." : "Save Operator Feedback"}
          </button>
        </div>
      </div>
    </div>
  );

  // Portal to document.body — completely outside the map/page DOM tree
  return createPortal(modal, document.body);
}

// ─── MapView ──────────────────────────────────────────────────────────────────
export function MapView() {
  const queryClient = useQueryClient();

  const [selectedMission, setSelectedMission] = useState<string>("");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("ALL");
  const [selectedLocationSource, setSelectedLocationSource] = useState<string>("ALL");
  const [selectedDetection, setSelectedDetection] = useState<DetectionRecord | null>(null);

  // Single source of truth for review modal
  const [reviewModalDetection, setReviewModalDetection] = useState<DetectionRecord | null>(null);
  const [reviewStatus, setReviewStatus] = useState<string>("confirmed");
  const [reviewNote, setReviewNote] = useState<string>("");

  // Leaflet map instance — populated by MapPopupCloser
  const leafletMapRef = useRef<LeafletMap | null>(null);
  const handleMapReady = useCallback((map: LeafletMap) => {
    leafletMapRef.current = map;
  }, []);

  const { data: missions } = useQuery({ queryKey: ["missions"], queryFn: getMissions });

  const { data: rawDetections, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["map-detections", selectedMission],
    queryFn: () => getAllDetections(selectedMission || undefined, 1000),
  });

  const reviewMutation = useMutation({
    mutationFn: ({ detectionId, status, note }: { detectionId: string; status: string; note: string }) =>
      reviewDetection(detectionId, { status, note }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["map-detections"] });
      queryClient.invalidateQueries({ queryKey: ["dashboard-stats"] });
      queryClient.invalidateQueries({ queryKey: ["recent-detections"] });
      queryClient.invalidateQueries({ queryKey: ["detections"] });
      setReviewModalDetection(null);
      setReviewNote("");
    },
  });

  // handleOpenReview:
  //   1. close Leaflet popup via map instance
  //   2. clear selected-marker state
  //   3. open review modal
  const handleOpenReview = useCallback((detection: DetectionRecord) => {
    if (leafletMapRef.current) {
      leafletMapRef.current.closePopup();
    }
    setSelectedDetection(null);
    setReviewModalDetection(detection);
    setReviewStatus(detection.operator_status || "confirmed");
    setReviewNote((detection as DetectionRecord & { operator_note?: string }).operator_note || "");
  }, []);

  const handleCloseModal = useCallback(() => {
    setReviewModalDetection(null);
    setReviewNote("");
  }, []);

  const handleSaveReview = useCallback(() => {
    if (!reviewModalDetection) return;
    reviewMutation.mutate({ detectionId: reviewModalDetection.detection_id, status: reviewStatus, note: reviewNote });
  }, [reviewModalDetection, reviewMutation, reviewStatus, reviewNote]);

  // ── Filtering ──────────────────────────────────────────────────────────────
  const realGeolocated = useMemo(() => {
    if (!rawDetections) return [];
    return rawDetections.filter((d) => {
      if (d.lat == null || d.lon == null) return false;
      if (isNaN(d.lat) || isNaN(d.lon) || d.lat < -90 || d.lat > 90 || d.lon < -180 || d.lon > 180) return false;
      const src = d.location_source || d.coordinates_label || "UNAVAILABLE";
      return src !== "UNAVAILABLE" && src !== "SIMULATED";
    });
  }, [rawDetections]);

  const isDemoFallback =
    realGeolocated.length === 0 &&
    selectedLocationSource !== "REAL_GPS" &&
    selectedLocationSource !== "SONAR_METADATA";

  const activeTargetsPool = isDemoFallback
    ? SIMULATED_OFFSHORE_TARGETS
    : realGeolocated.length > 0 ? realGeolocated : (rawDetections || []);

  const filteredDetections = useMemo(() => {
    return activeTargetsPool.filter((d) => {
      if (d.lat == null || d.lon == null) return false;
      if (selectedSeverity !== "ALL" && d.severity !== selectedSeverity) return false;
      if (selectedLocationSource !== "ALL") {
        const src = d.location_source || d.coordinates_label || "UNAVAILABLE";
        if (src !== selectedLocationSource) return false;
      }
      return true;
    });
  }, [activeTargetsPool, selectedSeverity, selectedLocationSource]);

  const mapCenter: [number, number] = useMemo(() => {
    if (selectedDetection?.lat && selectedDetection?.lon) return [selectedDetection.lat, selectedDetection.lon];
    if (filteredDetections.length > 0 && filteredDetections[0].lat && filteredDetections[0].lon)
      return [filteredDetections[0].lat, filteredDetections[0].lon];
    return [13.1500, 80.4850];
  }, [filteredDetections, selectedDetection]);

  const modalOpen = reviewModalDetection !== null;
  const highCount = filteredDetections.filter((d) => d.severity === "HIGH").length;
  const medCount  = filteredDetections.filter((d) => d.severity === "MEDIUM").length;
  const lowCount  = filteredDetections.filter((d) => d.severity === "LOW").length;

  return (
    <div className="space-y-4">
      {/* Demo Banner */}
      {isDemoFallback && (
        <div className="flex items-center gap-2.5 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-2.5 text-xs text-amber-900 shadow-2xs">
          <AlertCircle className="h-4 w-4 shrink-0 text-amber-700" />
          <span>
            <b>Demo Mode:</b> No verified spatial telemetry is available. Displayed targets are
            simulated for interface demonstration in offshore marine waters.
          </span>
        </div>
      )}

      {/* Header & Filters */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold text-text-navy flex items-center gap-2">
            <Compass className="h-5 w-5 text-ocean" /> Geospatial Telemetry Map
          </h1>
          <p className="text-xs text-text-secondary">
            Underwater acoustic target coordinates, hazard spatial distribution, and survey bounds.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2.5">
          <div className="flex items-center gap-1.5 bg-card border border-border rounded-lg px-2.5 py-1 text-xs shadow-2xs">
            <span className="text-text-secondary font-medium">Mission:</span>
            <select value={selectedMission} onChange={(e) => setSelectedMission(e.target.value)}
              className="bg-transparent text-text-navy font-semibold focus:outline-none cursor-pointer">
              <option value="">All Missions</option>
              {missions?.map((m) => <option key={m.mission_id} value={m.mission_id}>{m.name || m.mission_id}</option>)}
            </select>
          </div>
          <div className="flex items-center gap-1.5 bg-card border border-border rounded-lg px-2.5 py-1 text-xs shadow-2xs">
            <span className="text-text-secondary font-medium">Severity:</span>
            <select value={selectedSeverity} onChange={(e) => setSelectedSeverity(e.target.value)}
              className="bg-transparent text-text-navy font-semibold focus:outline-none cursor-pointer">
              <option value="ALL">All Severities</option>
              <option value="HIGH">High Risk</option>
              <option value="MEDIUM">Medium Risk</option>
              <option value="LOW">Low Risk</option>
            </select>
          </div>
          <div className="flex items-center gap-1.5 bg-card border border-border rounded-lg px-2.5 py-1 text-xs shadow-2xs">
            <span className="text-text-secondary font-medium">Source:</span>
            <select value={selectedLocationSource} onChange={(e) => setSelectedLocationSource(e.target.value)}
              className="bg-transparent text-text-navy font-semibold focus:outline-none cursor-pointer">
              <option value="ALL">All Sources</option>
              <option value="REAL_GPS">Real GPS</option>
              <option value="SONAR_METADATA">Sonar Telemetry</option>
              <option value="SIMULATED">Simulated</option>
              <option value="MANUAL">Manual</option>
            </select>
          </div>
        </div>
      </div>

      {/* Map + Side List */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Map wrapper — pointer-events:none while modal open (blocks all map interaction) */}
        <div
          className="lg:col-span-3 h-[600px] rounded-xl border border-border bg-card overflow-hidden shadow-xs relative"
          style={modalOpen ? { pointerEvents: "none", userSelect: "none" } : undefined}
        >
          {isLoading ? (
            <div className="h-full w-full flex items-center justify-center bg-bg-primary">
              <Skeleton className="h-full w-full" />
            </div>
          ) : isError ? (
            <div className="p-8 h-full flex items-center justify-center">
              <ErrorState error={error} onRetry={() => refetch()} />
            </div>
          ) : (
            <MapContainer center={mapCenter} zoom={12} scrollWheelZoom={true} className="h-full w-full">
              {/* MapPopupCloser: registers map instance for popup-close control */}
              <MapPopupCloser onMapReady={handleMapReady} />
              <MapRecenter center={mapCenter} />
              <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              {filteredDetections.map((d, index) => {
                if (!d.lat || !d.lon) return null;
                const isSelected = selectedDetection?.detection_id === d.detection_id;
                const isSimulated = (d.location_source || d.coordinates_label) === "SIMULATED";
                const color = d.severity === "HIGH" ? "#D9534F" : d.severity === "MEDIUM" ? "#D99000" : "#159A72";
                return (
                  <Marker
                    key={d.detection_id}
                    position={[d.lat, d.lon]}
                    icon={createCustomIcon(color, isSelected, isSimulated, index)}
                    eventHandlers={{ click: () => setSelectedDetection(d) }}
                  >
                    <Popup>
                      <div className="p-1 space-y-1.5 text-xs min-w-[210px]">
                        <div className="flex items-center justify-between font-bold text-text-navy border-b pb-1">
                          <span>{CLASS_LABELS[d.class_name] || d.class_name.replace(/_/g, " ").toUpperCase()}</span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                            d.severity === "HIGH" ? "bg-red-100 text-red-700"
                            : d.severity === "MEDIUM" ? "bg-amber-100 text-amber-700"
                            : "bg-emerald-100 text-emerald-700"}`}>
                            {d.severity}
                          </span>
                        </div>
                        <div className="font-mono text-[11px] text-gray-600">Detection: {d.detection_id}</div>
                        <div className="grid grid-cols-2 gap-1 text-[11px]">
                          <div>YOLO Conf: <span className="font-semibold">{Math.round(d.confidence * 100)}%</span></div>
                          <div>Evidence: <span className="font-semibold">
                            {d.evidence_score != null ? `${Math.round(d.evidence_score * 100)}%` : "N/A"}
                          </span></div>
                        </div>
                        <div className="text-[11px] text-gray-600">
                          Location: <span className="font-mono">{d.lat.toFixed(5)}, {d.lon.toFixed(5)}</span>
                        </div>
                        <div className="text-[11px] text-gray-600">
                          Location Source:{" "}
                          <span className={`font-semibold ${isSimulated ? "text-amber-700 font-mono" : "text-ocean"}`}>
                            {d.location_source || "GPS"}
                          </span>
                          {d.depth_m != null && ` | Depth: ${d.depth_m}m`}
                        </div>
                        {isSimulated && (
                          <div className="rounded bg-amber-50 border border-amber-200 px-2 py-1 text-[10px] font-medium text-amber-800">
                            Demo coordinate — not field GPS
                          </div>
                        )}
                        {/* Review Detection button: closes popup then opens modal */}
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); handleOpenReview(d); }}
                          className="mt-2 w-full rounded bg-ocean px-2 py-1.5 text-center text-xs font-semibold text-white hover:bg-ocean/90 transition-colors cursor-pointer"
                        >
                          Review Detection
                        </button>
                      </div>
                    </Popup>
                  </Marker>
                );
              })}
            </MapContainer>
          )}

          {/* Map Legend — z-[1000], below modal (9999) */}
          <div className="absolute bottom-4 left-4 z-[1000] rounded-lg bg-card/95 backdrop-blur-md border border-border p-3 shadow-md text-xs space-y-1.5">
            <div className="font-semibold text-text-navy flex items-center gap-1">
              <Layers className="h-3.5 w-3.5 text-ocean" /> Map Legend
            </div>
            <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#D9534F] border border-white" /><span className="text-text-navy">High Risk Hazard</span></div>
            <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#D99000] border border-white" /><span className="text-text-navy">Medium Priority Target</span></div>
            <div className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full bg-[#159A72] border border-white" /><span className="text-text-navy">Low Priority Target</span></div>
            <div className="flex items-center gap-2 pt-1 border-t border-border">
              <span className="h-2.5 w-2.5 rounded-full border-2 border-dashed border-gray-400 bg-transparent" />
              <span className="text-text-navy">Simulated Demo</span>
            </div>
          </div>

          {/* Target count badge — z-[1000] */}
          <div className="absolute top-3 right-3 z-[1000] rounded-lg bg-card/95 backdrop-blur-md border border-border px-3 py-2 shadow-md text-xs">
            <div className="font-semibold text-text-navy mb-1">
              <Crosshair className="inline h-3.5 w-3.5 text-ocean mr-1" />
              {filteredDetections.length} Target{filteredDetections.length !== 1 ? "s" : ""}
            </div>
            <div className="flex gap-2 text-[11px]">
              <span className="text-red-600 font-semibold">{highCount}H</span>
              <span className="text-amber-600 font-semibold">{medCount}M</span>
              <span className="text-emerald-600 font-semibold">{lowCount}L</span>
            </div>
          </div>
        </div>

        {/* Side Target List */}
        <div className="lg:col-span-1 flex flex-col gap-3">
          <div className="rounded-xl border border-border bg-card p-3 shadow-xs">
            <h2 className="text-sm font-semibold text-text-navy mb-2">Detected Targets</h2>
            <div className="space-y-2 max-h-[540px] overflow-y-auto pr-1">
              {filteredDetections.length === 0 ? (
                <div className="text-xs text-text-secondary text-center py-8">No targets match current filters.</div>
              ) : (
                filteredDetections.map((d, i) => {
                  const isSelected = selectedDetection?.detection_id === d.detection_id;
                  const isSim = (d.location_source || d.coordinates_label) === "SIMULATED";
                  return (
                    <div
                      key={d.detection_id}
                      style={{ animationDelay: `${Math.min(i * 30, 240)}ms` }}
                      className={`w-full p-3 rounded-lg border transition-all text-xs cursor-pointer animate-[pageEntrance_150ms_ease-out_both] ${
                        isSelected ? "border-ocean bg-ocean/10 shadow-2xs" : "border-border bg-bg-primary/50 hover:bg-bg-secondary"
                      }`}
                      onClick={() => setSelectedDetection(d)}
                    >
                      <div className="flex items-center justify-between font-semibold text-text-navy">
                        <span>{CLASS_LABELS[d.class_name] || d.class_name.replace(/_/g, " ")}</span>
                        <StatusBadge kind="severity" value={d.severity} />
                      </div>
                      <div className="font-mono text-[11px] text-text-secondary mt-1">
                        {d.lat?.toFixed(5)}, {d.lon?.toFixed(5)}{d.depth_m != null ? ` | ${d.depth_m}m` : ""}
                      </div>
                      <div className="text-[11px] text-text-secondary mt-1 flex items-center justify-between">
                        <div className="flex items-center gap-1.5">
                          <span>{Math.round(d.confidence * 100)}% conf</span>
                          {isSim && (
                            <span className="px-1.5 rounded font-bold text-[9px] bg-amber-100 text-amber-800 border border-amber-200">SIMULATED</span>
                          )}
                        </div>
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); handleOpenReview(d); }}
                          className="text-[10px] font-semibold text-ocean hover:underline"
                        >
                          Review
                        </button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Selected Detection Detail Card */}
          {selectedDetection && !modalOpen && (
            <div className="rounded-xl border border-ocean/30 bg-ocean/5 p-3 shadow-xs text-xs space-y-2 animate-[pageEntrance_150ms_ease-out]">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-text-navy">
                  {CLASS_LABELS[selectedDetection.class_name] || selectedDetection.class_name.replace(/_/g, " ")}
                </span>
                <StatusBadge kind="severity" value={selectedDetection.severity} />
              </div>
              <div className="font-mono text-[11px] text-text-secondary">{selectedDetection.detection_id}</div>
              <div className="grid grid-cols-2 gap-1 text-[11px]">
                <div>Conf: <span className="font-bold text-ocean">{Math.round(selectedDetection.confidence * 100)}%</span></div>
                <div>Evidence: <span className="font-bold text-text-navy">
                  {selectedDetection.evidence_score != null ? `${Math.round(selectedDetection.evidence_score * 100)}%` : "N/A"}
                </span></div>
              </div>
              <button
                type="button"
                onClick={() => handleOpenReview(selectedDetection)}
                className="w-full rounded-lg bg-ocean px-2 py-1.5 text-center text-xs font-semibold text-white hover:bg-ocean/90 transition-colors"
              >
                Review Detection
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Detection Review Modal — portal to document.body, z-index 9999/10000
          Rendered OUTSIDE map/page DOM. Above all Leaflet layers. */}
      {modalOpen && (
        <DetectionReviewModal
          detection={reviewModalDetection!}
          reviewStatus={reviewStatus}
          reviewNote={reviewNote}
          isSaving={reviewMutation.isPending}
          onStatusChange={setReviewStatus}
          onNoteChange={setReviewNote}
          onSave={handleSaveReview}
          onClose={handleCloseModal}
        />
      )}
    </div>
  );
}
