import type { PredictResponse } from "@/lib/api";
import { labelColor, labelVi, labelBg, labelSoftBg, labelRing } from "@/lib/labels";
import { ProbaBars } from "./ProbaBars";

export function VerdictCard({ result }: { result: PredictResponse }) {
  const confidence = Math.round(Math.max(...result.proba) * 100);
  return (
    <div className={`surface overflow-hidden ring-2 ${labelRing(result.label)}`}>
      <div className="flex">
        {/* Signature: severity signal bar runs the full height of the card. */}
        <div className={`w-1.5 shrink-0 ${labelBg(result.label)}`} aria-hidden />
        <div className="flex-1 p-5">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <div className="eyebrow mb-1">Phán quyết</div>
              <div className={`text-3xl font-bold leading-none ${labelColor(result.label)}`}>
                {result.label_name}
              </div>
              <div className="mt-1 text-sm text-slate-500">{labelVi(result.label)}</div>
            </div>
            <div className={`rounded-xl px-3 py-2 text-right ${labelSoftBg(result.label)}`}>
              <div className={`font-mono text-2xl font-semibold tnum ${labelColor(result.label)}`}>
                {confidence}%
              </div>
              <div className="text-[0.7rem] uppercase tracking-wide text-slate-500">Tin cậy</div>
            </div>
          </div>
          <ProbaBars proba={result.proba} />
          <div className="mt-4 text-xs text-slate-400">
            Mô hình: <span className="font-mono text-slate-500">{result.model}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
