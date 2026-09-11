import { useEffect, useRef, useState } from "react";
import type { Detection } from "../../api/types";

interface SonarViewerProps {
  imageUrl: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
  isScanning?: boolean;
}

const SEVERITY_STROKE: Record<string, string> = {
  HIGH: "#D9534F",
  MEDIUM: "#D99000",
  LOW: "#159A72",
  UNKNOWN: "#60758A",
};

export function SonarViewer({
  imageUrl,
  detections,
  imageWidth,
  imageHeight,
  isScanning = false,
}: SonarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [displayWidth, setDisplayWidth] = useState(0);
  const [revealedCount, setRevealedCount] = useState(0);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width;
      if (width) setDisplayWidth(width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Sequential Staggered Bounding Box Reveal (~60ms stagger for first 10 detections)
  useEffect(() => {
    if (detections.length === 0) {
      setRevealedCount(0);
      return;
    }

    setRevealedCount(1);
    const totalToStagger = Math.min(detections.length, 10);
    let count = 1;

    const interval = setInterval(() => {
      count++;
      setRevealedCount(count);
      if (count >= totalToStagger) {
        setRevealedCount(detections.length);
        clearInterval(interval);
      }
    }, 60);

    return () => clearInterval(interval);
  }, [detections]);

  // Draw bounding boxes on Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !displayWidth || !imageWidth) return;

    const scale = displayWidth / imageWidth;
    const displayHeight = imageHeight * scale;
    canvas.width = displayWidth;
    canvas.height = displayHeight;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const visibleDetections = detections.slice(0, revealedCount);

    for (const det of visibleDetections) {
      const [x1, y1, x2, y2] = det.bbox;
      const color = SEVERITY_STROKE[det.severity] ?? SEVERITY_STROKE.UNKNOWN;

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1 * scale, y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);

      const label = `${det.display_name} ${det.confidence_pct.toFixed(0)}%`;
      ctx.font = "600 11px Inter, sans-serif";
      const textWidth = ctx.measureText(label).width;
      const labelY = Math.max(y1 * scale - 18, 14);

      ctx.fillStyle = color;
      ctx.fillRect(x1 * scale, labelY - 12, textWidth + 8, 16);
      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(label, x1 * scale + 4, labelY);
    }
  }, [displayWidth, detections, revealedCount, imageWidth, imageHeight]);

  return (
    <div ref={containerRef} className="relative w-full overflow-hidden rounded-lg border border-border bg-bg-secondary">
      {/* Sonar Image with short crossfade transition */}
      <img
        src={imageUrl}
        alt="Sonar analysis result"
        onLoad={() => setIsLoaded(true)}
        className={`block w-full transition-opacity duration-150 ease-out ${isLoaded ? "opacity-100" : "opacity-70"}`}
      />

      {/* Real-time Bounding Box Canvas Overlay */}
      <canvas ref={canvasRef} className="pointer-events-none absolute left-0 top-0" />

      {/* Active AI Scan Line (Only active while analysis is running) */}
      {isScanning && (
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-color-ocean to-transparent shadow-[0_0_8px_rgba(8,126,164,0.7)] animate-[scanLine_2s_linear_infinite]" />
        </div>
      )}
    </div>
  );
}
