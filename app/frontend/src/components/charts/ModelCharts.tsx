"use client";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";
import type { InsightsResponse, ModelResult } from "@/lib/api";

const SEVERITY = ["#10b981", "#f59e0b", "#ef4444"]; // CLEAN / OFFENSIVE / HATE

const AXIS = { fontSize: 11, fill: "#64748b" };

/** Horizontal bars: confidence per model, tinted by predicted label. */
export function ConfidenceBarChart({ models }: { models: ModelResult[] }) {
  const data = models.map((m) => ({
    name: m.display_name,
    conf: Math.round(Math.max(...m.proba) * 100),
    label: m.label,
  }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <XAxis type="number" domain={[0, 100]} unit="%" tick={AXIS} />
        <YAxis type="category" dataKey="name" width={120} tick={AXIS} />
        <Tooltip
          cursor={{ fill: "rgba(15,23,42,0.04)" }}
          formatter={(v: number) => [`${v}%`, "Tin cậy"]}
        />
        <Bar dataKey="conf" radius={[0, 4, 4, 0]} barSize={16}>
          {data.map((d, i) => (
            <Cell key={i} fill={SEVERITY[d.label] ?? "#94a3b8"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Latency per model — shows the speed/accuracy trade-off. */
export function LatencyBarChart({ models }: { models: ModelResult[] }) {
  const data = models.map((m) => ({ name: m.display_name, ms: m.latency_ms }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(160, data.length * 34)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
        <XAxis type="number" unit="ms" tick={AXIS} />
        <YAxis type="category" dataKey="name" width={120} tick={AXIS} />
        <Tooltip
          cursor={{ fill: "rgba(15,23,42,0.04)" }}
          formatter={(v: number) => [`${v} ms`, "Độ trễ"]}
        />
        <Bar dataKey="ms" fill="#4f46e5" radius={[0, 4, 4, 0]} barSize={16} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** All 7 models compared on Accuracy and F1_macro (the macro gap exposes
 *  who actually handles the minority HATE class, not just the easy majority). */
export function MetricsBarChart({ data }: { data: InsightsResponse }) {
  const rows = data.models.map((m) => ({
    name: m.display_name,
    Accuracy: m.accuracy,
    F1_macro: m.f1_macro,
    best: m.display_name === data.best,
  }));
  return (
    <ResponsiveContainer width="100%" height={Math.max(220, rows.length * 42)}>
      <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 16 }} barGap={2}>
        <XAxis type="number" domain={[0, 1]} tick={AXIS} />
        <YAxis type="category" dataKey="name" width={120} tick={AXIS} />
        <Tooltip
          cursor={{ fill: "rgba(15,23,42,0.04)" }}
          formatter={(v: number) => v.toFixed(4)}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Accuracy" fill="#4f46e5" radius={[0, 3, 3, 0]} barSize={9} />
        <Bar dataKey="F1_macro" fill="#10b981" radius={[0, 3, 3, 0]} barSize={9} />
      </BarChart>
    </ResponsiveContainer>
  );
}
