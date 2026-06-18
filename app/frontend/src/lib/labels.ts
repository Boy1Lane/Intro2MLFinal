export const LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"] as const;

const VI = ["Sạch", "Xúc phạm", "Thù ghét"];
const TEXT = ["text-emerald-600", "text-amber-600", "text-red-600"];
const BG = ["bg-emerald-500", "bg-amber-500", "bg-red-500"];

export function labelVi(label: number): string {
  return VI[label] ?? "?";
}
export function labelColor(label: number): string {
  return TEXT[label] ?? "text-slate-600";
}
export function labelBg(label: number): string {
  return BG[label] ?? "bg-slate-400";
}
