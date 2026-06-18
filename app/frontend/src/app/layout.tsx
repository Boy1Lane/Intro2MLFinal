import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "ViHSD Moderation Studio",
  description: "Phát hiện ngôn từ thù ghét tiếng Việt",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>
        <Providers>
          <header className="border-b bg-white">
            <nav className="mx-auto flex max-w-5xl items-center gap-6 px-4 py-3">
              <Link href="/" className="font-semibold text-slate-900">🛡️ ViHSD Studio</Link>
              <Link href="/simulate" className="text-sm text-slate-600 hover:text-slate-900">Mô phỏng</Link>
              <Link href="/insights" className="text-sm text-slate-600 hover:text-slate-900">Kết quả mô hình</Link>
            </nav>
          </header>
          <main className="mx-auto max-w-5xl px-4 py-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
