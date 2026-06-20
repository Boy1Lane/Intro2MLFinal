"use client";
import { useState } from "react";

const SAMPLES = [
  "Cảm ơn bạn nhiều nhé, bài viết rất hữu ích",
  "Nói chuyện vô duyên thế, phiền phức quá",
  "Lũ người đó đáng bị tiêu diệt hết",
];

export function InputBar({ onAnalyze, loading }: { onAnalyze: (t: string) => void; loading: boolean }) {
  const [text, setText] = useState("");
  const empty = text.trim().length === 0;
  return (
    <div className="surface p-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        maxLength={5000}
        aria-label="Bình luận cần phân tích"
        placeholder="Nhập bình luận tiếng Việt..."
        className="w-full resize-none rounded-xl border border-slate-200 bg-white p-3 text-slate-800 outline-none transition-colors focus:border-indigo-400 focus:ring-2 focus:ring-indigo-500/20"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <span className="text-xs text-slate-400">Ví dụ:</span>
        {SAMPLES.map((s, i) => (
          <button
            key={i}
            onClick={() => setText(s)}
            title={s}
            className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600 transition-colors hover:bg-slate-200"
          >
            {["Sạch", "Xúc phạm", "Thù ghét"][i]}
          </button>
        ))}
        <span className="ml-auto font-mono text-xs tnum text-slate-300">{text.length}/5000</span>
        <button
          onClick={() => onAnalyze(text)}
          disabled={loading || empty}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-indigo-700 disabled:opacity-40 disabled:hover:bg-indigo-600"
        >
          {loading ? "Đang phân tích..." : "Phân tích"}
        </button>
      </div>
    </div>
  );
}
