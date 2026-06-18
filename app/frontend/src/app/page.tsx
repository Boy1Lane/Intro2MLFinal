"use client";
import { useMutation } from "@tanstack/react-query";
import { predict, showdown, rewrite } from "@/lib/api";
import { InputBar } from "@/components/InputBar";
import { VerdictCard } from "@/components/VerdictCard";
import { ExplainPanel } from "@/components/ExplainPanel";
import { ShowdownTable } from "@/components/ShowdownTable";
import { RewriteCard } from "@/components/RewriteCard";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";
import { useState } from "react";

export default function StudioPage() {
  const [current, setCurrent] = useState("");
  const predictM = useMutation({ mutationFn: predict });
  const showdownM = useMutation({ mutationFn: showdown });
  const rewriteM = useMutation({ mutationFn: rewrite });

  function analyze(text: string) {
    setCurrent(text);
    rewriteM.reset();
    predictM.mutate(text);
    showdownM.mutate(text);
  }

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-slate-900">Comment Moderation Studio</h1>
      <InputBar onAnalyze={analyze} loading={predictM.isPending} />

      {predictM.isError && <ErrorNote message={(predictM.error as Error).message} />}
      {predictM.isPending && <Spinner label="Đang chấm điểm..." />}
      {predictM.data && (
        <>
          <VerdictCard result={predictM.data} />
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <h3 className="mb-3 font-semibold text-slate-800">Vì sao? (giải thích theo mô hình tuyến tính)</h3>
            <ExplainPanel tokens={predictM.data.tokens} />
          </div>
        </>
      )}

      {showdownM.isPending && <Spinner label="Đang chạy 7 mô hình..." />}
      {showdownM.data && <ShowdownTable models={showdownM.data.models} />}

      {predictM.data && (
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <button
            onClick={() => rewriteM.mutate(current)}
            disabled={rewriteM.isPending}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {rewriteM.isPending ? "Đang viết lại..." : "Viết lại lịch sự"}
          </button>
          {rewriteM.isError && <div className="mt-3"><ErrorNote message={(rewriteM.error as Error).message} /></div>}
          {rewriteM.data && <div className="mt-4"><RewriteCard data={rewriteM.data} /></div>}
        </div>
      )}
    </div>
  );
}
