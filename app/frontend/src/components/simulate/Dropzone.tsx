"use client";
import { UploadCloud } from "lucide-react";

export function Dropzone({ onFile, disabled }: { onFile: (f: File) => void; disabled?: boolean }) {
  return (
    <label className="flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 bg-white p-8 text-slate-500 hover:border-slate-400">
      <UploadCloud className="h-8 w-8" />
      <span className="text-sm">Tải lên file CSV (cột <code>free_text</code>)</span>
      <input
        type="file"
        accept=".csv"
        disabled={disabled}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
      />
    </label>
  );
}
