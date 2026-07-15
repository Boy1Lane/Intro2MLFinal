"use client";
import { UploadCloud } from "lucide-react";

export function Dropzone({ onFile, disabled }: { onFile: (f: File) => void; disabled?: boolean }) {
  return (
    <div className="space-y-2">
      <label className="flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 bg-white/70 p-10 text-slate-500 transition-colors hover:border-indigo-400 hover:bg-indigo-50/40 hover:text-indigo-600">
        <UploadCloud className="h-8 w-8" />
        <span className="text-sm">Tải lên file CSV</span>
        <input
          type="file"
          accept=".csv"
          disabled={disabled}
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
        />
      </label>
      <div className="rounded-xl border border-slate-200 bg-slate-50/70 px-4 py-3 text-xs text-slate-500">
        <p className="font-medium text-slate-600">Định dạng CSV mong đợi</p>
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          <li>Bắt buộc có cột <code className="font-mono text-slate-700">free_text</code> (mỗi dòng một bình luận).</li>
          <li>Các cột khác (ví dụ <code className="font-mono text-slate-700">label_id</code>) được bỏ qua — có thể nộp thẳng <code className="font-mono text-slate-700">test.csv</code>.</li>
          <li>Mã hóa UTF-8; tối đa 5.000 dòng mỗi lần.</li>
        </ul>
      </div>
    </div>
  );
}
