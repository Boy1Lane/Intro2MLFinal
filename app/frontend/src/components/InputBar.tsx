"use client";
import { useState } from "react";

const SAMPLES = [
  "Cảm ơn bạn nhiều nhé, bài viết rất hữu ích",
  "Nói chuyện vô duyên thế, phiền phức quá",
  "Lũ người đó đáng bị tiêu diệt hết",
];

export function InputBar({ onAnalyze, loading }: { onAnalyze: (t: string) => void; loading: boolean }) {
  const [text, setText] = useState("");
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        aria-label="Bình luận cần phân tích"
        placeholder="Nhập bình luận tiếng Việt..."
        className="w-full resize-none rounded-lg border border-slate-200 p-3 text-slate-800 outline-none focus:border-slate-400"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {SAMPLES.map((s, i) => (
          <button key={i} onClick={() => setText(s)}
            className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600 hover:bg-slate-200">
            Ví dụ {i + 1}
          </button>
        ))}
        <button
          onClick={() => onAnalyze(text)}
          disabled={loading || text.trim().length === 0}
          className="ml-auto rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {loading ? "Đang phân tích..." : "Phân tích"}
        </button>
      </div>
    </div>
  );
}
