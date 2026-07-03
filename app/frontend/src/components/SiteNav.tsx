import Link from "next/link";
import { Shield } from "lucide-react";

export function SiteNav() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/70 bg-white/70 backdrop-blur-md">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold text-slate-900">
          <span className="grid h-8 w-8 place-items-center rounded-xl bg-indigo-600 text-white shadow-sm">
            <Shield className="h-4 w-4" />
          </span>
          ViHSD Studio
        </Link>
        <nav className="flex items-center gap-4 text-sm text-slate-600">
          <Link href="/" className="hover:text-indigo-600">Studio</Link>
          <Link href="/monitor" className="hover:text-indigo-600">Monitor</Link>
        </nav>
      </div>
    </header>
  );
}
