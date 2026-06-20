import type { InsightsResponse } from "@/lib/api";

const COLS: { key: keyof Omit<InsightsResponse["models"][number], "display_name">; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision_w", label: "Precision_w" },
  { key: "recall_w", label: "Recall_w" },
  { key: "f1_w", label: "F1_w" },
  { key: "f1_macro", label: "F1_macro" },
];

export function MetricsTable({ data }: { data: InsightsResponse }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-[0.7rem] uppercase tracking-wide text-slate-400">
          <th className="pb-2 font-medium">Mô hình</th>
          {COLS.map((c) => (
            <th key={c.key} className="pb-2 font-medium">{c.label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.models.map((m) => {
          const isBest = m.display_name === data.best;
          return (
            <tr
              key={m.display_name}
              data-testid="metric-row"
              data-best={isBest ? "true" : undefined}
              className={`border-t border-slate-100 transition-colors ${
                isBest ? "bg-emerald-50/70 font-semibold" : "hover:bg-slate-50"
              }`}
            >
              <td className="py-2.5 text-slate-700">
                <span className="inline-flex items-center gap-2">
                  {m.display_name}
                  {isBest && (
                    <span className="rounded-full bg-emerald-600 px-1.5 py-0.5 text-[0.6rem] font-semibold uppercase tracking-wide text-white">
                      Best
                    </span>
                  )}
                </span>
              </td>
              {COLS.map((c) => (
                <td key={c.key} className="py-2.5 font-mono tnum text-slate-600">
                  {m[c.key].toFixed(4)}
                </td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
