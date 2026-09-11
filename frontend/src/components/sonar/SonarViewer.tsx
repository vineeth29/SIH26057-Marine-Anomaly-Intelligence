import { useEffect, useRef, useState } from "react";
import type { Detection } from "../../api/types";

/**
 * Renders a sonar image (any variant — pass the "raw" URL for the
 * interactive default view) with bounding boxes drawn client-side from
 * real backend detection coordinates (Detection.bbox), scaled to the
 * displayed image size. This is intentionally separate from the
 * server-baked "annotated" image variant, which exists as a static
 * fallback/download rather than what's shown here.
 */
interface SonarViewerProps {
  imageUrl: string;
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
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
}: SonarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [displayWidth, setDisplayWidth] = useState(0);

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

    for (const det of detections) {
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
  }, [displayWidth, detections, imageWidth, imageHeight]);

  return (
    <div ref={containerRef} className="relative w-full overflow-hidden rounded-lg border border-border bg-bg-secondary">
      {/* eslint-disable-next-line jsx-a11y/alt-text */}
      <img src={imageUrl} alt="Sonar analysis result" className="block w-full" />
      <canvas ref={canvasRef} className="pointer-events-none absolute left-0 top-0" />
    </div>
  );
}
