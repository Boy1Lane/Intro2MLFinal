export const LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"] as const;

const VI = ["Sạch", "Xúc phạm", "Thù ghét"];
const TEXT = ["text-emerald-600", "text-amber-600", "text-red-600"];
const BG = ["bg-emerald-500", "bg-amber-500", "bg-red-500"];
// Full literal class strings (no interpolation) so Tailwind keeps them.
const SOFT = ["bg-emerald-50", "bg-amber-50", "bg-red-50"];
const RING = ["ring-emerald-500/20", "ring-amber-500/20", "ring-red-500/20"];
const BORDER = ["border-emerald-500", "border-amber-500", "border-red-500"];

export function labelVi(label: number): string {
  return VI[label] ?? "?";
}
export function labelColor(label: number): string {
  return TEXT[label] ?? "text-slate-600";
}
export function labelBg(label: number): string {
  return BG[label] ?? "bg-slate-400";
}
export function labelSoftBg(label: number): string {
  return SOFT[label] ?? "bg-slate-100";
}
export function labelRing(label: number): string {
  return RING[label] ?? "ring-slate-500/20";
}
export function labelBorder(label: number): string {
  return BORDER[label] ?? "border-slate-400";
}
