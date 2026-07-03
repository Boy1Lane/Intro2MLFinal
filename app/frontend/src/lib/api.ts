const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Verdict { label: number; label_name: string; proba: number[]; }
export interface TokenScore { token: string; score: number; }
export interface PredictResponse extends Verdict { tokens: TokenScore[]; model: string; }
export interface ModelResult { name: string; display_name: string; label: number; proba: number[]; latency_ms: number; }
export interface ShowdownResponse { models: ModelResult[]; }
export interface RewriteResponse { rewritten: string; before: Verdict; after: Verdict; }
export interface BatchRow { text: string; label: number; label_name: string; proba: number[]; }
export interface BatchResponse { total: number; counts: Record<string, number>; toxic_ratio: number; rows: BatchRow[]; }
export interface ModelMetric { display_name: string; accuracy: number; precision_w: number; recall_w: number; f1_w: number; f1_macro: number; }
export interface InsightsResponse { best: string; models: ModelMetric[]; }

async function jsonPost<T>(path: string, text: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handle<T>(res);
}

async function handle<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (body as { detail?: string }).detail || `Lỗi ${res.status}`;
    throw new Error(detail);
  }
  return body as T;
}

export const predict = (text: string) => jsonPost<PredictResponse>("/predict", text);
export const showdown = (text: string) => jsonPost<ShowdownResponse>("/showdown", text);
export const rewrite = (text: string) => jsonPost<RewriteResponse>("/rewrite", text);

export async function batch(file: File): Promise<BatchResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/batch`, { method: "POST", body: fd });
  return handle<BatchResponse>(res);
}

export async function getInsights(): Promise<InsightsResponse> {
  return handle<InsightsResponse>(await fetch(`${BASE}/insights`));
}
export async function getHealth(): Promise<{ sklearn_loaded: boolean; phobert_available: boolean }> {
  return handle(await fetch(`${BASE}/health`));
}

export interface MonitorComment { text: string; label: number; label_name: string; proba: number[]; toxic: boolean; model: string; seen_at: string; tokens: TokenScore[]; }
export interface WatchSummary { id: string; url: string; label: string | null; created_at: string; last_scan: string | null; last_error: string | null; alert_count: number; total_comments: number; toxic_count: number; }
export interface WatchDetail extends WatchSummary { comments: MonitorComment[]; }

export async function listWatches(): Promise<WatchSummary[]> {
  return handle<WatchSummary[]>(await fetch(`${BASE}/monitor/watches`));
}
export async function createWatch(url: string, label?: string): Promise<WatchSummary> {
  const res = await fetch(`${BASE}/monitor/watches`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, label: label || null }),
  });
  return handle<WatchSummary>(res);
}
export async function getWatch(id: string): Promise<WatchDetail> {
  return handle<WatchDetail>(await fetch(`${BASE}/monitor/watches/${id}`));
}
export async function scanWatch(id: string): Promise<WatchDetail> {
  return handle<WatchDetail>(await fetch(`${BASE}/monitor/watches/${id}/scan`, { method: "POST" }));
}
export async function ackWatch(id: string): Promise<WatchSummary> {
  return handle<WatchSummary>(await fetch(`${BASE}/monitor/watches/${id}/ack`, { method: "POST" }));
}
export async function deleteWatch(id: string): Promise<void> {
  const res = await fetch(`${BASE}/monitor/watches/${id}`, { method: "DELETE" });
  await handle(res);
}
