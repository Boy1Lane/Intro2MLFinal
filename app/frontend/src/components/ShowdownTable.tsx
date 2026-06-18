import type { ModelResult } from "@/lib/api";
import { labelColor } from "@/lib/labels";

export function ShowdownTable({ models }: { models: ModelResult[] }) {
  const consensus = models.length > 0 && models.every((m) => m.label === models[0].label);
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-semibold text-slate-800">So sánh mô hình</h3>
        <span className={`rounded-full px-2 py-0.5 text-xs ${consensus ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
          {consensus ? "Đồng thuận" : "Bất đồng"}
        </span>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs uppercase text-slate-400">
            <th className="pb-2">Mô hình</th><th className="pb-2">Nhãn</th>
            <th className="pb-2">Tin cậy</th><th className="pb-2">Độ trễ</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const conf = Math.round(Math.max(...m.proba) * 100);
            return (
              <tr key={m.name} data-testid="showdown-row" className="border-t">
                <td className="py-2 font-medium text-slate-700">{m.display_name}</td>
                <td className={`py-2 font-semibold ${labelColor(m.label)}`}>{m.label}</td>
                <td className="py-2 tabular-nums text-slate-600">{conf}%</td>
                <td className="py-2 tabular-nums text-slate-400">{m.latency_ms} ms</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
