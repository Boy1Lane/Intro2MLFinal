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
        <tr className="text-left text-xs uppercase text-slate-400">
          <th className="pb-2">Mô hình</th>
          {COLS.map((c) => <th key={c.key} className="pb-2">{c.label}</th>)}
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
              className={`border-t ${isBest ? "bg-emerald-50 font-semibold" : ""}`}
            >
              <td className="py-2 text-slate-700">{m.display_name}</td>
              {COLS.map((c) => (
                <td key={c.key} className="py-2 tabular-nums text-slate-600">{m[c.key].toFixed(4)}</td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
