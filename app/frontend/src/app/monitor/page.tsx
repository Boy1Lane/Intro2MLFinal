"use client";
import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle, RefreshCw, Trash2, Check, Plus, ChevronLeft, ChevronRight,
} from "lucide-react";
import {
  listWatches, getMonitorModels, getMonitorSources, createWatch, getWatch,
  scanWatch, ackWatch, deleteWatch, type WatchSummary, type TokenScore,
} from "@/lib/api";
import { labelVi, labelColor, labelSoftBg, labelBorder } from "@/lib/labels";

const PAGE_SIZE = 8;

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
  const [selectedId, setSelectedId] = useState<string | null>(null);
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

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["watches"] });
    qc.invalidateQueries({ queryKey: ["watch"] });
  };

  const add = useMutation({
    mutationFn: () => createWatch(url.trim(), label.trim() || undefined, model),
    onSuccess: (created) => {
      setUrl(""); setLabel(""); setError(null); setSelectedId(created.id); refresh();
    },
    onError: (e: Error) => setError(e.message),
  });
  const scan = useMutation({ mutationFn: scanWatch, onSuccess: refresh, onError: (e: Error) => setError(e.message) });
  const ack = useMutation({ mutationFn: ackWatch, onSuccess: refresh, onError: (e: Error) => setError(e.message) });
  const remove = useMutation({
    mutationFn: deleteWatch,
    onSuccess: (_d, id) => { if (id === selectedId) setSelectedId(null); refresh(); },
    onError: (e: Error) => setError(e.message),
  });

  // drop a stale selection if its watch disappears
  if (selectedId && watches.data && !watches.data.some((w) => w.id === selectedId)) {
    setSelectedId(null);
  }

  return (
    <main className="mx-auto max-w-5xl px-4 py-8">
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
          className="flex-1 rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          placeholder="https://vnexpress.net/bai-viet-123.html"
          aria-label="URL cần theo dõi"
          value={url} onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && url.trim()) add.mutate(); }}
        />
        <input
          className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:w-36"
          placeholder="Tên (tuỳ chọn)"
          aria-label="Tên gợi nhớ"
          value={label} onChange={(e) => setLabel(e.target.value)}
        />
        <select
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 sm:w-44"
          value={model} onChange={(e) => setModel(e.target.value)}
          aria-label="Model phân loại"
        >
          {(models.data ?? [{ key: "PhoBERT", name: "PhoBERT-base-v2" }]).map((m) => (
            <option key={m.key} value={m.key}>{m.name}</option>
          ))}
        </select>
        <button
          className="inline-flex items-center justify-center gap-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 focus-visible:ring-offset-2 disabled:opacity-50"
          disabled={!url.trim() || add.isPending}
          onClick={() => add.mutate()}
        >
          <Plus className="h-4 w-4" /> {add.isPending ? "Đang thêm…" : "Thêm"}
        </button>
      </div>
      {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

      <div className="mt-6 grid gap-4 md:grid-cols-[minmax(0,340px)_1fr]">
        {/* Watch list */}
        <div className="space-y-2">
          {watches.isLoading && (
            <p className="text-sm text-slate-400">Đang tải danh sách theo dõi…</p>
          )}
          {watches.isError && (
            <p className="text-sm text-red-600">Không tải được danh sách. Thử lại sau.</p>
          )}
          {watches.data?.length === 0 && (
            <p className="text-sm text-slate-400">Chưa có URL nào được theo dõi.</p>
          )}
          {watches.data?.map((w) => (
            <WatchRow
              key={w.id} watch={w} modelLabel={modelName(w.model)}
              selected={w.id === selectedId}
              onSelect={() => setSelectedId(w.id)}
            />
          ))}
        </div>

        {/* Detail panel */}
        <div className="min-h-[16rem]">
          <AnimatePresence mode="wait">
            {selectedId ? (
              <motion.div
                key={selectedId}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -12 }}
                transition={{ duration: 0.22, ease: "easeOut" }}
              >
                <DetailPanel
                  watchId={selectedId}
                  onScan={() => scan.mutate(selectedId)}
                  onAck={() => ack.mutate(selectedId)}
                  onDelete={() => remove.mutate(selectedId)}
                  scanning={scan.isPending && scan.variables === selectedId}
                />
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="grid h-full min-h-[16rem] place-items-center rounded-2xl border border-dashed border-slate-200 text-sm text-slate-400"
              >
                Chọn một trang bên trái để xem chi tiết bình luận.
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </main>
  );
}

function WatchRow({ watch, modelLabel, selected, onSelect }: {
  watch: WatchSummary; modelLabel: string; selected: boolean; onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect} aria-pressed={selected}
      className={`w-full rounded-xl border p-3 text-left shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
        selected
          ? "border-indigo-400 bg-indigo-50"
          : "border-slate-200 bg-white hover:border-slate-300"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="min-w-0">
          <span className="block truncate font-medium text-slate-800">
            {watch.label || watch.url}
          </span>
          <span className="block truncate text-xs text-slate-400">{watch.url}</span>
        </span>
        {watch.alert_count > 0 && (
          <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-red-50 px-2 py-0.5 text-xs font-medium text-red-600">
            <AlertTriangle className="h-3 w-3" /> {watch.alert_count}
          </span>
        )}
      </div>
      <div className="mt-1.5 flex items-center gap-2 text-[10px] text-slate-400">
        <span className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-500">{modelLabel}</span>
        <span>{watch.total_comments} bình luận · {watch.toxic_count} tiêu cực</span>
      </div>
    </button>
  );
}

function DetailPanel({ watchId, onScan, onAck, onDelete, scanning }: {
  watchId: string; onScan: () => void; onAck: () => void;
  onDelete: () => void; scanning: boolean;
}) {
  const [page, setPage] = useState(1);
  const detail = useQuery({
    queryKey: ["watch", watchId],
    queryFn: () => getWatch(watchId),
    refetchInterval: 15000,
  });
  const w = detail.data;

  const sorted = (w?.comments ?? []).slice().sort(
    (a, b) => (b.toxic ? 1 : 0) - (a.toxic ? 1 : 0) || b.seen_at.localeCompare(a.seen_at),
  );
  const pages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const current = Math.min(page, pages);
  const shown = sorted.slice((current - 1) * PAGE_SIZE, current * PAGE_SIZE);
  const hasTokens = sorted.some((c) => c.toxic && c.tokens.length > 0);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate font-medium text-slate-800">{w?.label || w?.url || "…"}</p>
          {w?.label && <p className="truncate text-xs text-slate-400">{w.url}</p>}
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <IconButton label="Quét ngay" onClick={onScan}>
            <RefreshCw className={`h-4 w-4 ${scanning ? "animate-spin" : ""}`} />
          </IconButton>
          <IconButton label="Đánh dấu đã đọc" onClick={onAck}>
            <Check className="h-4 w-4" />
          </IconButton>
          <IconButton label="Xoá watch" onClick={onDelete} danger>
            <Trash2 className="h-4 w-4" />
          </IconButton>
        </div>
      </div>

      <p className="mt-2 text-xs text-slate-400">
        {w?.last_error
          ? <span className="text-red-500">Lỗi: {w.last_error}</span>
          : `Quét gần nhất: ${w?.last_scan ? new Date(w.last_scan).toLocaleString("vi-VN") : "chưa"} · ${w?.total_comments ?? 0} bình luận · ${w?.toxic_count ?? 0} tiêu cực`}
      </p>

      <div className="mt-3 space-y-2 border-t border-slate-100 pt-3">
        {detail.isLoading && <p className="text-xs text-slate-400">Đang tải bình luận…</p>}
        {w && sorted.length === 0 && (
          <p className="text-xs text-slate-400">Chưa trích được bình luận nào.</p>
        )}
        {hasTokens && (
          <p className="text-[10px] text-slate-400">
            <mark className="rounded bg-red-200 px-0.5 text-red-900">từ tô đỏ</mark>{" "}
            = từ khiến bình luận bị gắn nhãn tiêu cực
          </p>
        )}
        {shown.map((c) => {
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

        {pages > 1 && (
          <div className="flex items-center justify-between pt-1 text-xs text-slate-500">
            <button
              onClick={() => setPage(current - 1)} disabled={current <= 1}
              aria-label="Trang trước"
              className="inline-flex items-center gap-1 rounded-lg px-2 py-1 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-40"
            >
              <ChevronLeft className="h-4 w-4" /> Trước
            </button>
            <span>Trang {current} / {pages}</span>
            <button
              onClick={() => setPage(current + 1)} disabled={current >= pages}
              aria-label="Trang sau"
              className="inline-flex items-center gap-1 rounded-lg px-2 py-1 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 disabled:opacity-40"
            >
              Sau <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function IconButton({ label, onClick, danger, children }: {
  label: string; onClick: () => void; danger?: boolean; children: React.ReactNode;
}) {
  return (
    <button
      type="button" onClick={onClick} title={label} aria-label={label}
      className={`rounded-lg p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
        danger ? "text-red-500 hover:bg-red-50" : "text-slate-600 hover:bg-slate-100"
      }`}
    >
      {children}
    </button>
  );
}
