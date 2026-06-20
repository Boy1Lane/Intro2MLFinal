import { LABEL_NAMES, labelBg } from "@/lib/labels";

export function ProbaBars({ proba }: { proba: number[] }) {
  const top = proba.indexOf(Math.max(...proba));
  return (
    <div className="space-y-2">
      {proba.map((p, i) => (
        <div key={i} className="flex items-center gap-3">
          <span
            className={`w-24 text-xs font-medium ${i === top ? "text-slate-800" : "text-slate-500"}`}
          >
            {LABEL_NAMES[i]}
          </span>
          <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              data-testid="proba-bar"
              className={`h-2.5 rounded-full transition-[width] duration-500 ease-out ${labelBg(i)} ${
                i === top ? "" : "opacity-40"
              }`}
              style={{ width: `${Math.round(p * 100)}%` }}
            />
          </div>
          <span className="w-10 text-right font-mono text-xs tnum text-slate-500">
            {Math.round(p * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
