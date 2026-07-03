"use client";
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RefreshCw, Trash2, Check, Plus } from "lucide-react";
import {
  listWatches, getMonitorModels, getMonitorSources, createWatch, getWatch,
  scanWatch, ackWatch, deleteWatch, type WatchSummary, type TokenScore,
} from "@/lib/api";
import { labelVi, labelColor, labelSoftBg, labelBorder } from "@/lib/labels";

// Wrap the words that pushed a comment toward its toxic label. Explanation
// tokens are lowercased/normalized upstream, so match on a lowercased,
// punctuation-trimmed form of each original word (catches plain hate words).
const STRIP = /^[^0-9a-zà-ỹ]+|[^0-9a-zà-ỹ]+$/gi;
function highlight(text: string, tokens: TokenScore[]): React.ReactNode {
  const toxic = new Set(tokens.filter((t) => t.score > 0).map((t) => t.token.toLowerCase()));
  if (toxic.size === 0) return text;
  return text.split(/(\s+)/).map((part, i) => {
    if (/^\s+$/.test(part) || !part) return part;
    const norm = part.toLowerCase().replace(STRIP, "");
    return norm && toxic.has(norm) ? (
      <mark key={i} className="rounded bg-red-200 px-0.5 text-red-900">{part}</mark>
    ) : (
      part
    );
  });
}

export default function MonitorPage() {
  const qc = useQueryClient();
  const [url, setUrl] = useState("");
  const [label, setLabel] = useState("");
  const [model, setModel] = useState("PhoBERT");
  const [openId, setOpenId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const watches = useQuery({
    queryKey: ["watches"],
    queryFn: listWatches,
    refetchInterval: 15000,
  });
  const models = useQuery({ queryKey: ["monitor-models"], queryFn: getMonitorModels });
  const sources = useQuery({ queryKey: ["monitor-sources"], queryFn: getMonitorSources });
  const modelName = (key: string) =>
    models.data?.find((m) => m.key === key)?.name ?? key;

  const refresh = () => qc.invalidateQueries({ queryKey: ["watches"] });

  const add = useMutation({
    mutationFn: () => createWatch(url.trim(), label.trim() || undefined, model),
    onSuccess: () => { setUrl(""); setLabel(""); setError(null); refresh(); },
    onError: (e: Error) => setError(e.message),
  });
  const scan = useMutation({ mutationFn: scanWatch, onSuccess: refresh, onError: (e: Error) => setError(e.message) });
  const ack = useMutation({ mutationFn: ackWatch, onSuccess: refresh, onError: (e: Error) => setError(e.message) });
  const remove = useMutation({ mutationFn: deleteWatch, onSuccess: refresh, onError: (e: Error) => setError(e.message) });

  return (
    <main className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="text-2xl font-semibold text-slate-900">Monitor</h1>
      <p className="mt-1 text-sm text-slate-500">
        Theo dõi URL định kỳ (mỗi vài phút). Bình luận tiêu cực mới sẽ hiện cảnh báo đỏ.
      </p>
      {sources.data && sources.data.length > 0 && (
        <p className="mt-1 text-xs text-slate-400">
          Hỗ trợ tốt (lấy đúng bình luận):{" "}
          <span className="font-medium text-slate-500">
            {sources.data.map((s) => s.display).join(", ")}
          </span>{" "}
          · trang khác chạy best-effort.
        </p>
      )}

      <div className="mt-6 flex flex-col gap-2 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:flex-row">
        <input
          className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm"
          placeholder="https://trang-tin.com/bai-viet"
          value={url} onChange={(e) => setUrl(e.target.value)}
        />
        <input
          className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm sm:w-36"
          placeholder="Tên (tuỳ chọn)"
          value={label} onChange={(e) => setLabel(e.target.value)}
        />
        <select
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm sm:w-44"
          value={model} onChange={(e) => setModel(e.target.value)}
          title="Model phân loại"
        >
          {(models.data ?? [{ key: "PhoBERT", name: "PhoBERT-base-v2" }]).map((m) => (
            <option key={m.key} value={m.key}>{m.name}</option>
          ))}
        </select>
        <button
          className="inline-flex items-center justify-center gap-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          disabled={!url.trim() || add.isPending}
          onClick={() => add.mutate()}
        >
          <Plus className="h-4 w-4" /> Thêm
        </button>
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

      <div className="mt-6 space-y-3">
        {watches.data?.length === 0 && (
          <p className="text-sm text-slate-400">Chưa có URL nào được theo dõi.</p>
        )}
        {watches.data?.map((w) => (
          <WatchCard
            key={w.id} watch={w} modelLabel={modelName(w.model)}
            open={openId === w.id}
            onToggle={() => setOpenId(openId === w.id ? null : w.id)}
            onScan={() => scan.mutate(w.id)}
            onAck={() => ack.mutate(w.id)}
            onDelete={() => remove.mutate(w.id)}
            scanning={scan.isPending && scan.variables === w.id}
          />
        ))}
      </div>
    </main>
  );
}

function WatchCard({ watch, modelLabel, open, onToggle, onScan, onAck, onDelete, scanning }: {
  watch: WatchSummary; modelLabel: string; open: boolean; onToggle: () => void;
  onScan: () => void; onAck: () => void; onDelete: () => void; scanning: boolean;
}) {
  const detail = useQuery({
    queryKey: ["watch", watch.id],
    queryFn: () => getWatch(watch.id),
    enabled: open,
    refetchInterval: open ? 15000 : false,
  });

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <button className="min-w-0 flex-1 text-left" onClick={onToggle}>
          <p className="truncate font-medium text-slate-800">
            {watch.label || watch.url}
          </p>
          <p className="truncate text-xs text-slate-400">{watch.url}</p>
          <span className="mt-0.5 inline-block rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
            {modelLabel}
          </span>
        </button>
        <div className="flex items-center gap-2">
          {watch.alert_count > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-1 text-xs font-medium text-red-600">
              <AlertTriangle className="h-3 w-3" /> {watch.alert_count}
            </span>
          )}
          <button title="Quét ngay" onClick={onScan} className="rounded-lg p-2 hover:bg-slate-100">
            <RefreshCw className={`h-4 w-4 ${scanning ? "animate-spin" : ""}`} />
          </button>
          <button title="Đánh dấu đã đọc" onClick={onAck} className="rounded-lg p-2 hover:bg-slate-100">
            <Check className="h-4 w-4" />
          </button>
          <button title="Xoá" onClick={onDelete} className="rounded-lg p-2 text-red-500 hover:bg-red-50">
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      <p className="mt-2 text-xs text-slate-400">
        {watch.last_error
          ? <span className="text-red-500">Lỗi: {watch.last_error}</span>
          : `Quét gần nhất: ${watch.last_scan ? new Date(watch.last_scan).toLocaleString("vi-VN") : "chưa"} · ${watch.total_comments} bình luận · ${watch.toxic_count} tiêu cực`}
      </p>

      {open && (
        <div className="mt-3 space-y-2 border-t border-slate-100 pt-3">
          {detail.data?.comments.length === 0 && (
            <p className="text-xs text-slate-400">Chưa trích được bình luận nào.</p>
          )}
          {detail.data?.comments.some((c) => c.toxic && c.tokens.length > 0) && (
            <p className="text-[10px] text-slate-400">
              <mark className="rounded bg-red-200 px-0.5 text-red-900">từ tô đỏ</mark>{" "}
              = từ khiến bình luận bị gắn nhãn tiêu cực
            </p>
          )}
          {detail.data?.comments
            .slice()
            // toxic first, then newest first within each group
            .sort((a, b) =>
              (b.toxic ? 1 : 0) - (a.toxic ? 1 : 0) || b.seen_at.localeCompare(a.seen_at))
            .map((c) => {
              const pct = Math.round((c.proba[c.label] ?? 0) * 100);
              return (
                <div
                  key={`${c.seen_at}-${c.text}`}
                  className={`rounded-lg p-2 text-sm ${labelSoftBg(c.label)} ${
                    c.toxic ? `border-l-4 ${labelBorder(c.label)}` : "opacity-60"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className={`inline-flex items-center gap-1 text-xs font-semibold ${labelColor(c.label)}`}>
                      {c.toxic && <AlertTriangle className="h-3 w-3" />}
                      {labelVi(c.label)} · {pct}%
                    </span>
                    <span className="text-[10px] text-slate-400">{c.model}</span>
                  </div>
                  <p className="mt-1 text-slate-700">
                    {c.toxic && c.tokens.length > 0 ? highlight(c.text, c.tokens) : c.text}
                  </p>
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}
