"use client";
import { UploadCloud } from "lucide-react";

export function Dropzone({ onFile, disabled }: { onFile: (f: File) => void; disabled?: boolean }) {
  return (
    <label className="flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 bg-white/70 p-10 text-slate-500 transition-colors hover:border-indigo-400 hover:bg-indigo-50/40 hover:text-indigo-600">
      <UploadCloud className="h-8 w-8" />
      <span className="text-sm">Tải lên file CSV (cột <code className="font-mono text-slate-700">free_text</code>)</span>
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
