"use client";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import type { RewriteResponse } from "@/lib/api";
import { labelColor, labelVi } from "@/lib/labels";

export function RewriteCard({ data }: { data: RewriteResponse }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl border bg-white p-5 shadow-sm"
    >
      <h3 className="mb-2 font-semibold text-slate-800">Gợi ý viết lại lịch sự</h3>
      <p className="mb-4 rounded-lg bg-slate-50 p-3 text-slate-700">{data.rewritten}</p>
      <div className="flex items-center gap-3 text-sm">
        <span className={`font-semibold ${labelColor(data.before.label)}`}>{labelVi(data.before.label)}</span>
        <ArrowRight className="h-4 w-4 text-slate-400" />
        <span className={`font-semibold ${labelColor(data.after.label)}`}>{labelVi(data.after.label)}</span>
      </div>
    </motion.div>
  );
}
