"use client";
import { useMutation } from "@tanstack/react-query";
import { batch } from "@/lib/api";
import { Dropzone } from "@/components/simulate/Dropzone";
import { BatchDashboard } from "@/components/simulate/BatchDashboard";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";

export default function SimulatePage() {
  const m = useMutation({ mutationFn: batch });
  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-slate-900">Mô phỏng kiểm duyệt hàng loạt</h1>
      <Dropzone onFile={(f) => m.mutate(f)} disabled={m.isPending} />
      {m.isPending && <Spinner label="Đang chấm điểm file..." />}
      {m.isError && <ErrorNote message={(m.error as Error).message} />}
      {m.data && <BatchDashboard data={m.data} />}
    </div>
  );
}
