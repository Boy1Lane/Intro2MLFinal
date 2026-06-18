import type { TokenScore } from "@/lib/api";

function chipStyle(score: number, max: number): string {
  if (score === 0 || max === 0) return "bg-slate-100 text-slate-600";
  const intensity = Math.min(1, Math.abs(score) / max);
  const level = intensity > 0.66 ? 200 : intensity > 0.33 ? 100 : 50;
  return score > 0
    ? `bg-red-${level} text-red-800`
    : `bg-emerald-${level} text-emerald-800`;
}

export function ExplainPanel({ tokens }: { tokens: TokenScore[] }) {
  if (tokens.length === 0) {
    return <p className="text-sm text-slate-400">Không có token để giải thích.</p>;
  }
  const max = Math.max(...tokens.map((t) => Math.abs(t.score)), 0);
  return (
    <div className="flex flex-wrap gap-1.5">
      {tokens.map((t, i) => (
        <span
          key={i}
          title={t.score.toFixed(3)}
          className={`rounded px-2 py-1 text-sm ${chipStyle(t.score, max)}`}
        >
          {t.token}
        </span>
      ))}
    </div>
  );
}
