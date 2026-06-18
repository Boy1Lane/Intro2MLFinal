import { LABEL_NAMES, labelBg } from "@/lib/labels";

export function ProbaBars({ proba }: { proba: number[] }) {
  return (
    <div className="space-y-2">
      {proba.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-24 text-xs font-medium text-slate-600">{LABEL_NAMES[i]}</span>
          <div className="h-3 flex-1 rounded-full bg-slate-200">
            <div
              data-testid="proba-bar"
              className={`h-3 rounded-full ${labelBg(i)}`}
              style={{ width: `${Math.round(p * 100)}%` }}
            />
          </div>
          <span className="w-10 text-right text-xs tabular-nums text-slate-500">
            {Math.round(p * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
