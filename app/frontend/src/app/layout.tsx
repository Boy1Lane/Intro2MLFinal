import "./globals.css";
import type { Metadata } from "next";
import { Be_Vietnam_Pro, IBM_Plex_Mono } from "next/font/google";
import { Providers } from "./providers";
import { SiteNav } from "@/components/SiteNav";

const sans = Be_Vietnam_Pro({
  subsets: ["latin", "vietnamese"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-sans",
  display: "swap",
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ViHSD Moderation Studio",
  description: "Phát hiện ngôn từ thù ghét tiếng Việt — phân loại CLEAN / OFFENSIVE / HATE",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi" className={`${sans.variable} ${mono.variable}`}>
      <body className="font-sans">
        <Providers>
          <SiteNav />
          <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
          <footer className="mx-auto max-w-6xl px-4 pb-10 pt-4 text-xs text-slate-400">
            ViHSD Moderation Studio · 7 mô hình tiếng Việt (6 sklearn + PhoBERT-base-v2) ·
            Kết quả mang tính hỗ trợ kiểm duyệt, không thay thế quyết định của con người.
          </footer>
        </Providers>
      </body>
    </html>
  );
}
