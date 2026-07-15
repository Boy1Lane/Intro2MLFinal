"use client";
import { useMutation, useQuery } from "@tanstack/react-query";
import { predict, showdown, rewrite, batch, getInsights } from "@/lib/api";
import { InputBar } from "@/components/InputBar";
import { VerdictCard } from "@/components/VerdictCard";
import { ExplainPanel } from "@/components/ExplainPanel";
import { ShowdownTable } from "@/components/ShowdownTable";
import { RewriteCard } from "@/components/RewriteCard";
import { Dropzone } from "@/components/simulate/Dropzone";
import { BatchDashboard } from "@/components/simulate/BatchDashboard";
import { MetricsTable } from "@/components/insights/MetricsTable";
import {
  ConfidenceBarChart,
  LatencyBarChart,
  MetricsBarChart,
} from "@/components/charts/ModelCharts";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";
import { Sparkles, MessageSquareText, FileText, Award } from "lucide-react";
import { useState } from "react";

type Mode = "single" | "batch";

export default function StudioPage() {
  const [mode, setMode] = useState<Mode>("single");
  const [current, setCurrent] = useState("");
  const predictM = useMutation({ mutationFn: predict });
  const showdownM = useMutation({ mutationFn: showdown });
  const rewriteM = useMutation({ mutationFn: rewrite });
  const batchM = useMutation({ mutationFn: batch });
  const insights = useQuery({ queryKey: ["insights"], queryFn: getInsights });

  function analyze(text: string) {
    setCurrent(text);
    rewriteM.reset();
    predictM.mutate(text);
    showdownM.mutate(text);
  }

  const idle = !predictM.data && !predictM.isPending && !predictM.isError;

  return (
    <div className="space-y-10">
      {/* Hero */}
      <header>
        <p className="eyebrow mb-1">Comment Moderation Studio</p>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">
          Kiểm duyệt bình luận tiếng Việt
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-500">
          Chấm điểm một bình luận hoặc cả file CSV bằng 7 mô hình (6 sklearn + PhoBERT-base-v2):
          phán quyết, mức tin cậy, lý do và bản viết lại lịch sự.
        </p>
      </header>

      {/* Analyze */}
      <section id="analyze" className="scroll-mt-20 space-y-6">
        <div className="inline-flex rounded-xl border border-slate-200 bg-white p-1 shadow-sm">
          <ModeTab active={mode === "single"} onClick={() => setMode("single")} icon={<MessageSquareText className="h-4 w-4" />}>
            Một bình luận
          </ModeTab>
          <ModeTab active={mode === "batch"} onClick={() => setMode("batch")} icon={<FileText className="h-4 w-4" />}>
            Tải CSV
          </ModeTab>
        </div>

        {mode === "single" ? (
          <div className="space-y-6">
            <InputBar onAnalyze={analyze} loading={predictM.isPending} />

            {idle && (
              <div className="surface flex flex-col items-center gap-2 px-6 py-14 text-center">
                <MessageSquareText className="h-7 w-7 text-slate-300" />
                <p className="text-sm text-slate-500">
                  Nhập bình luận hoặc chọn một ví dụ phía trên để bắt đầu.
                </p>
              </div>
            )}

            {predictM.isError && <ErrorNote message={(predictM.error as Error).message} />}
            {predictM.isPending && <Spinner label="Đang chấm điểm..." />}

            {predictM.data && (
              <div className="grid gap-6 lg:grid-cols-2">
                <VerdictCard result={predictM.data} />
                <section className="surface p-5">
                  <h2 className="eyebrow mb-3">Vì sao — đóng góp của từng token</h2>
                  <ExplainPanel tokens={predictM.data.tokens} />
                  <p className="mt-4 text-xs text-slate-400">
                    Đỏ = đẩy về phía độc hại, xanh = đẩy về phía sạch (mô hình tuyến tính).
                  </p>
                </section>
              </div>
            )}

            {showdownM.isPending && <Spinner label="Đang chạy 7 mô hình..." />}
            {showdownM.isError && <ErrorNote message={(showdownM.error as Error).message} />}
            {showdownM.data && (
              <div className="grid gap-6 lg:grid-cols-2">
                <ShowdownTable models={showdownM.data.models} />
                <div className="surface p-5">
                  <h2 className="eyebrow mb-3">Độ tin cậy theo mô hình</h2>
                  <ConfidenceBarChart models={showdownM.data.models} />
                  <h2 className="eyebrow mb-3 mt-5">Độ trễ suy luận</h2>
                  <LatencyBarChart models={showdownM.data.models} />
                </div>
              </div>
            )}

            {predictM.data && (
              <section className="surface p-5">
                <button
                  onClick={() => rewriteM.mutate(current)}
                  disabled={rewriteM.isPending}
                  className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-emerald-700 disabled:opacity-40"
                >
                  <Sparkles className="h-4 w-4" />
                  {rewriteM.isPending ? "Đang viết lại..." : "Viết lại lịch sự"}
                </button>
                {rewriteM.isError && (
                  <div className="mt-3">
                    <ErrorNote message={(rewriteM.error as Error).message} />
                  </div>
                )}
                {rewriteM.data && (
                  <div className="mt-4">
                    <RewriteCard data={rewriteM.data} />
                  </div>
                )}
              </section>
            )}
          </div>
        ) : (
          <div className="space-y-6">
            <Dropzone onFile={(f) => batchM.mutate(f)} disabled={batchM.isPending} />
            {batchM.isPending && <Spinner label="Đang chấm điểm file..." />}
            {batchM.isError && <ErrorNote message={(batchM.error as Error).message} />}
            {batchM.data && <BatchDashboard data={batchM.data} />}
          </div>
        )}
      </section>

      {/* Static offline benchmark, visually separated from the live analysis above */}
      <section id="models" className="scroll-mt-20 space-y-4">
        <div className="relative py-1">
          <div className="absolute inset-0 flex items-center" aria-hidden>
            <div className="w-full border-t-2 border-slate-300" />
          </div>
          <div className="relative flex justify-center">
            <span className="bg-white px-4 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Benchmark · Tập Test ViHSD
            </span>
          </div>
        </div>
        <h2 className="text-2xl font-bold tracking-tight text-slate-900">Hiệu năng 7 mô hình</h2>
        {insights.isPending && <Spinner label="Đang tải..." />}
        {insights.isError && <ErrorNote message={(insights.error as Error).message} />}
        {insights.data && (
          <>
            <div className="flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50/60 px-4 py-3 text-sm text-emerald-800">
              <Award className="h-4 w-4 shrink-0" />
              Mô hình tốt nhất: <span className="font-semibold">{insights.data.best}</span>
            </div>
            <div className="grid gap-6 lg:grid-cols-5">
              <div className="surface overflow-x-auto p-5 lg:col-span-3">
                <MetricsTable data={insights.data} />
              </div>
              <div className="surface p-5 lg:col-span-2">
                <h3 className="eyebrow mb-3">Accuracy vs F1_macro — cả 7 mô hình</h3>
                <MetricsBarChart data={insights.data} />
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  );
}

function ModeTab({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
        active ? "bg-slate-900 text-white shadow-sm" : "text-slate-600 hover:bg-slate-100"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}
