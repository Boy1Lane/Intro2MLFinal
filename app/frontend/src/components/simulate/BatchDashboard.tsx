"use client";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import type { BatchResponse } from "@/lib/api";
import { labelColor } from "@/lib/labels";

const COLORS = ["#10b981", "#f59e0b", "#ef4444"];

export function BatchDashboard({ data }: { data: BatchResponse }) {
  const pie = [
    { name: "CLEAN", value: data.counts.CLEAN ?? 0 },
    { name: "OFFENSIVE", value: data.counts.OFFENSIVE ?? 0 },
    { name: "HATE", value: data.counts.HATE ?? 0 },
  ];
  const topToxic = [...data.rows].sort((a, b) => b.proba[2] - a.proba[2]).slice(0, 20);

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <Stat label="Tổng bình luận" value={String(data.total)} />
        <Stat label="Tỉ lệ độc hại" value={`${Math.round(data.toxic_ratio * 100)}%`} />
        <Stat label="HATE" value={String(data.counts.HATE ?? 0)} />
      </div>
      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <h3 className="mb-2 font-semibold text-slate-800">Phân bố nhãn</h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={pie} dataKey="value" nameKey="name" innerRadius={50} outerRadius={80}>
                {pie.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <h3 className="mb-3 font-semibold text-slate-800">Bình luận độc hại nhất</h3>
        <ul className="space-y-1 text-sm">
          {topToxic.map((r, i) => (
            <li key={i} data-testid="toxic-row" className="flex items-center justify-between gap-3 border-t py-2">
              <span className="truncate text-slate-700">{r.text}</span>
              <span className={`shrink-0 font-semibold ${labelColor(r.label)}`}>{Math.round(r.proba[2] * 100)}%</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <div className="text-2xl font-bold text-slate-900">{value}</div>
      <div className="text-xs text-slate-500">{label}</div>
    </div>
  );
}
