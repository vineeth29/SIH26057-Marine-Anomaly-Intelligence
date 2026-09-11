import { useCallback, useRef, useState } from "react";
import { UploadCloud, FileImage } from "lucide-react";
import { cn } from "../../utils/cn";

interface UploadPanelProps {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
}

export function UploadPanel({ onFileSelected, selectedFile }: UploadPanelProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed px-6 py-10 text-center transition-all duration-150 ease-out",
        isDragging
          ? "border-ocean bg-ocean/5 scale-[1.01]"
          : "border-border bg-bg-secondary/40 scale-100"
      )}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      {selectedFile ? (
        <>
          <FileImage className="h-6 w-6 text-ocean" strokeWidth={1.75} />
          <div className="text-sm font-medium text-text-navy">{selectedFile.name}</div>
          <div className="text-xs text-text-secondary">
            {(selectedFile.size / 1024).toFixed(0)} KB • click Browse to replace
          </div>
        </>
      ) : (
        <>
          <UploadCloud className="h-6 w-6 text-text-secondary" strokeWidth={1.5} />
          <div className="text-sm font-medium text-text-navy">
            Drag & drop sonar image
          </div>
          <div className="text-xs text-text-secondary">PNG or JPEG</div>
        </>
      )}

      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        className="mt-2 rounded-md border border-border bg-card px-3 py-1.5 text-xs font-medium text-text-navy hover:bg-bg-secondary transition-colors cursor-pointer"
      >
        Browse Files
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
