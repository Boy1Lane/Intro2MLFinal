"use client";
import { useQuery } from "@tanstack/react-query";
import { getInsights } from "@/lib/api";
import { MetricsTable } from "@/components/insights/MetricsTable";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";

export default function InsightsPage() {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["insights"],
    queryFn: getInsights,
  });

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-slate-900">Kết quả mô hình (test set)</h1>
      {isPending && <Spinner label="Đang tải..." />}
      {isError && <ErrorNote message={(error as Error).message} />}
      {data && (
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          <MetricsTable data={data} />
        </div>
      )}
    </div>
  );
}
