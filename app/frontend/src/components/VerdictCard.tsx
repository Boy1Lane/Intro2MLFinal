import type { PredictResponse } from "@/lib/api";
import { labelColor, labelVi } from "@/lib/labels";
import { ProbaBars } from "./ProbaBars";

export function VerdictCard({ result }: { result: PredictResponse }) {
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-baseline justify-between">
        <div>
          <div className={`text-3xl font-bold ${labelColor(result.label)}`}>{result.label_name}</div>
          <div className="text-sm text-slate-500">{labelVi(result.label)}</div>
        </div>
        <div className="text-xs text-slate-400">Mô hình: {result.model}</div>
      </div>
      <ProbaBars proba={result.proba} />
    </div>
  );
}
