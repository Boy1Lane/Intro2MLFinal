# ViHSD Frontend (Next.js) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Next.js "Comment Moderation Studio" frontend that consumes the ViHSD backend — a single-page Studio pipeline (predict → explain → showdown → rewrite), a batch CSV dashboard, and a static insights page.

**Architecture:** Next.js App Router (TS) with a thin typed fetch client (`lib/api.ts`) mirroring the backend response shapes. React Query handles request state (loading/error). Pages compose focused presentational components; each component renders one section of the pipeline. Styling is Tailwind; charts use Recharts; motion uses framer-motion; icons use lucide-react.

**Tech Stack:** Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, @tanstack/react-query, recharts, framer-motion, lucide-react, Vitest + @testing-library/react + jsdom (component tests).

## Global Constraints

- All frontend code under `app/frontend/`. Node 18+.
- Backend base URL from `process.env.NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). Never hardcode a deployed URL.
- Labels are fixed and ordered `[CLEAN, OFFENSIVE, HATE]` (indices 0,1,2). Colors: CLEAN=emerald/green, OFFENSIVE=amber/orange, HATE=red. Vietnamese UI copy.
- Response types must match the backend exactly:
  - `POST /predict {text}` → `{label:number, label_name:string, proba:number[3], tokens:{token:string,score:number}[], model:string}`
  - `POST /showdown {text}` → `{models:{name:string, display_name:string, label:number, proba:number[3], latency_ms:number}[]}`
  - `POST /rewrite {text}` → `{rewritten:string, before:Verdict, after:Verdict}` where `Verdict={label:number,label_name:string,proba:number[3]}`
  - `POST /batch` (multipart `file`) → `{total:number, counts:{CLEAN:number,OFFENSIVE:number,HATE:number}, toxic_ratio:number, rows:{text:string,label:number,label_name:string,proba:number[3]}[]}`
  - `GET /insights` → `{best:string, models:{display_name:string,accuracy:number,precision_w:number,recall_w:number,f1_w:number,f1_macro:number}[]}`
- DEVIATION from design spec: use hand-rolled Tailwind components (not the shadcn CLI) for deterministic agentic builds. Visual polish ("đẹp nhất") is applied during implementation; structure here is functional-first.
- Every network call shows a loading state and surfaces errors (no silent failures).
- All paths relative to the worktree root `app/frontend/`.

## Backend dependency

The backend (separate plan, already implemented) must be running at `NEXT_PUBLIC_API_URL` for the app to function at runtime. Component tests mock `fetch`/the api client and do NOT require a live backend.

## File Structure

```
app/frontend/
  package.json
  tsconfig.json
  next.config.mjs
  postcss.config.mjs
  tailwind.config.ts
  vitest.config.ts
  vitest.setup.ts
  .env.local.example
  src/
    lib/
      api.ts            # typed client + TS types for all endpoints
      labels.ts         # LABEL_NAMES, label color classes, helpers
    app/
      globals.css
      layout.tsx        # root layout, nav, React Query provider
      providers.tsx     # QueryClientProvider
      page.tsx          # "/" Studio
      simulate/page.tsx # "/simulate"
      insights/page.tsx # "/insights"
    components/
      InputBar.tsx
      VerdictCard.tsx
      ProbaBars.tsx
      ExplainPanel.tsx
      ShowdownTable.tsx
      RewriteCard.tsx
      Spinner.tsx
      ErrorNote.tsx
      simulate/Dropzone.tsx
      simulate/BatchDashboard.tsx
      insights/MetricsTable.tsx
  tests/
    api.test.ts
    labels.test.ts
    verdict-card.test.tsx
    showdown-table.test.tsx
    explain-panel.test.tsx
    metrics-table.test.tsx
```

---

### Task 1: Scaffold project + tooling + lib (api client, labels)

**Files:**
- Create: `app/frontend/package.json`, `tsconfig.json`, `next.config.mjs`, `postcss.config.mjs`, `tailwind.config.ts`, `vitest.config.ts`, `vitest.setup.ts`, `.env.local.example`
- Create: `app/frontend/src/lib/labels.ts`, `app/frontend/src/lib/api.ts`, `app/frontend/src/app/globals.css`
- Test: `app/frontend/tests/labels.test.ts`, `app/frontend/tests/api.test.ts`

**Interfaces:**
- Produces (TS types, all `export`ed from `lib/api.ts`): `Verdict`, `PredictResponse`, `ModelResult`, `ShowdownResponse`, `RewriteResponse`, `BatchResponse`, `InsightsResponse`.
- Produces (functions in `lib/api.ts`): `predict(text)`, `showdown(text)`, `rewrite(text)`, `batch(file)`, `getInsights()`, `getHealth()` — all async, throwing `Error` with a readable message on non-2xx.
- Produces (`lib/labels.ts`): `LABEL_NAMES: string[]` = `["CLEAN","OFFENSIVE","HATE"]`; `labelColor(label:number): string` (Tailwind text color class); `labelBg(label:number): string` (Tailwind bg class); `labelVi(label:number): string` (Vietnamese gloss).

- [ ] **Step 1: Create package.json**

`app/frontend/package.json`:
```json
{
  "name": "vihsd-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "vitest run"
  },
  "dependencies": {
    "next": "14.2.5",
    "react": "18.3.1",
    "react-dom": "18.3.1",
    "@tanstack/react-query": "5.51.1",
    "recharts": "2.12.7",
    "framer-motion": "11.3.2",
    "lucide-react": "0.412.0"
  },
  "devDependencies": {
    "typescript": "5.5.3",
    "@types/react": "18.3.3",
    "@types/react-dom": "18.3.0",
    "@types/node": "20.14.10",
    "tailwindcss": "3.4.6",
    "postcss": "8.4.39",
    "autoprefixer": "10.4.19",
    "vitest": "2.0.3",
    "@testing-library/react": "16.0.0",
    "@testing-library/jest-dom": "6.4.6",
    "jsdom": "24.1.0",
    "@vitejs/plugin-react": "4.3.1"
  }
}
```

- [ ] **Step 2: Create config files**

`app/frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": false,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`app/frontend/next.config.mjs`:
```js
/** @type {import('next').NextConfig} */
const nextConfig = {};
export default nextConfig;
```

`app/frontend/postcss.config.mjs`:
```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`app/frontend/tailwind.config.ts`:
```ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
export default config;
```

`app/frontend/vitest.config.ts`:
```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    setupFiles: ["./vitest.setup.ts"],
    globals: true,
  },
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
});
```

`app/frontend/vitest.setup.ts`:
```ts
import "@testing-library/jest-dom/vitest";
```

`app/frontend/.env.local.example`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

- [ ] **Step 3: Create globals.css**

`app/frontend/src/app/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root { color-scheme: light; }
body { @apply bg-slate-50 text-slate-900 antialiased; }
```

- [ ] **Step 4: Write the failing tests**

`app/frontend/tests/labels.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { LABEL_NAMES, labelColor, labelBg, labelVi } from "@/lib/labels";

describe("labels", () => {
  it("has the three ordered labels", () => {
    expect(LABEL_NAMES).toEqual(["CLEAN", "OFFENSIVE", "HATE"]);
  });
  it("maps each label to distinct color classes", () => {
    const colors = [labelColor(0), labelColor(1), labelColor(2)];
    expect(new Set(colors).size).toBe(3);
    expect(labelBg(2)).toContain("red");
  });
  it("gives Vietnamese glosses", () => {
    expect(labelVi(0)).toBe("Sạch");
    expect(labelVi(2)).toBe("Thù ghét");
  });
});
```

`app/frontend/tests/api.test.ts`:
```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { predict, showdown, batch } from "@/lib/api";

beforeEach(() => { vi.restoreAllMocks(); });

function mockFetch(body: unknown, ok = true, status = 200) {
  return vi.spyOn(globalThis, "fetch").mockResolvedValue({
    ok, status,
    json: async () => body,
  } as Response);
}

describe("api client", () => {
  it("predict posts text and returns typed body", async () => {
    const f = mockFetch({ label: 2, label_name: "HATE", proba: [0.1, 0.2, 0.7], tokens: [], model: "PhoBERT-base-v2" });
    const res = await predict("xấu");
    expect(res.label).toBe(2);
    expect(f).toHaveBeenCalledWith(
      expect.stringContaining("/predict"),
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("showdown returns models array", async () => {
    mockFetch({ models: [{ name: "PhoBERT", display_name: "PhoBERT-base-v2", label: 0, proba: [0.9, 0.05, 0.05], latency_ms: 12 }] });
    const res = await showdown("ok");
    expect(res.models[0].display_name).toBe("PhoBERT-base-v2");
  });

  it("throws a readable error on non-2xx", async () => {
    mockFetch({ detail: "Text rỗng." }, false, 400);
    await expect(predict("")).rejects.toThrow("Text rỗng.");
  });

  it("batch sends FormData", async () => {
    const f = mockFetch({ total: 0, counts: { CLEAN: 0, OFFENSIVE: 0, HATE: 0 }, toxic_ratio: 0, rows: [] });
    const file = new File(["free_text\nhi"], "c.csv", { type: "text/csv" });
    await batch(file);
    const call = f.mock.calls[0][1] as RequestInit;
    expect(call.body).toBeInstanceOf(FormData);
  });
});
```

- [ ] **Step 5: Install deps and run tests to verify they fail**

Run: `cd app/frontend && npm install`
Run: `cd app/frontend && npm run test`
Expected: FAIL — `Cannot find module '@/lib/labels'` / `'@/lib/api'`.

- [ ] **Step 6: Implement labels.ts**

`app/frontend/src/lib/labels.ts`:
```ts
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
```

- [ ] **Step 7: Implement api.ts**

`app/frontend/src/lib/api.ts`:
```ts
const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Verdict { label: number; label_name: string; proba: number[]; }
export interface TokenScore { token: string; score: number; }
export interface PredictResponse extends Verdict { tokens: TokenScore[]; model: string; }
export interface ModelResult { name: string; display_name: string; label: number; proba: number[]; latency_ms: number; }
export interface ShowdownResponse { models: ModelResult[]; }
export interface RewriteResponse { rewritten: string; before: Verdict; after: Verdict; }
export interface BatchRow { text: string; label: number; label_name: string; proba: number[]; }
export interface BatchResponse { total: number; counts: Record<string, number>; toxic_ratio: number; rows: BatchRow[]; }
export interface ModelMetric { display_name: string; accuracy: number; precision_w: number; recall_w: number; f1_w: number; f1_macro: number; }
export interface InsightsResponse { best: string; models: ModelMetric[]; }

async function jsonPost<T>(path: string, text: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return handle<T>(res);
}

async function handle<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = (body as { detail?: string }).detail || `Lỗi ${res.status}`;
    throw new Error(detail);
  }
  return body as T;
}

export const predict = (text: string) => jsonPost<PredictResponse>("/predict", text);
export const showdown = (text: string) => jsonPost<ShowdownResponse>("/showdown", text);
export const rewrite = (text: string) => jsonPost<RewriteResponse>("/rewrite", text);

export async function batch(file: File): Promise<BatchResponse> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${BASE}/batch`, { method: "POST", body: fd });
  return handle<BatchResponse>(res);
}

export async function getInsights(): Promise<InsightsResponse> {
  return handle<InsightsResponse>(await fetch(`${BASE}/insights`));
}
export async function getHealth(): Promise<{ sklearn_loaded: boolean; phobert_available: boolean }> {
  return handle(await fetch(`${BASE}/health`));
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd app/frontend && npm run test`
Expected: PASS (labels + api suites green).

- [ ] **Step 9: Commit**

```bash
git add app/frontend/package.json app/frontend/package-lock.json app/frontend/*.ts app/frontend/*.mjs app/frontend/*.json app/frontend/.env.local.example app/frontend/src/lib app/frontend/src/app/globals.css app/frontend/tests/labels.test.ts app/frontend/tests/api.test.ts
git commit -m "feat(frontend): scaffold Next.js + typed api client + labels"
```

(Note: add `app/frontend/node_modules/` and `app/frontend/.next/` to the repo `.gitignore` before committing — do NOT track them.)

---

### Task 2: Root layout, providers, shared UI primitives

**Files:**
- Create: `app/frontend/src/app/providers.tsx`, `app/frontend/src/app/layout.tsx`
- Create: `app/frontend/src/components/Spinner.tsx`, `app/frontend/src/components/ErrorNote.tsx`, `app/frontend/src/components/ProbaBars.tsx`
- Test: `app/frontend/tests/proba-bars.test.tsx`

**Interfaces:**
- Consumes: `labelBg`, `LABEL_NAMES` from `lib/labels`.
- Produces: `<Providers>` (wraps children in a `QueryClientProvider`); `RootLayout`; `<Spinner/>`; `<ErrorNote message/>`; `<ProbaBars proba={number[]}/>` (renders 3 labelled bars whose widths are `proba[i]*100%`, each tagged with `data-testid="proba-bar"`).

- [ ] **Step 1: Write the failing test**

`app/frontend/tests/proba-bars.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProbaBars } from "@/components/ProbaBars";

describe("ProbaBars", () => {
  it("renders one bar per class with percentage labels", () => {
    render(<ProbaBars proba={[0.7, 0.2, 0.1]} />);
    const bars = screen.getAllByTestId("proba-bar");
    expect(bars).toHaveLength(3);
    expect(screen.getByText("CLEAN")).toBeInTheDocument();
    expect(screen.getByText("70%")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npm run test -- proba-bars`
Expected: FAIL — cannot find `@/components/ProbaBars`.

- [ ] **Step 3: Implement ProbaBars**

`app/frontend/src/components/ProbaBars.tsx`:
```tsx
import { LABEL_NAMES, labelBg } from "@/lib/labels";

export function ProbaBars({ proba }: { proba: number[] }) {
  return (
    <div className="space-y-2">
      {proba.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-24 text-xs font-medium text-slate-600">{LABEL_NAMES[i]}</span>
          <div className="h-3 flex-1 rounded-full bg-slate-200">
            <div
              data-testid="proba-bar"
              className={`h-3 rounded-full ${labelBg(i)}`}
              style={{ width: `${Math.round(p * 100)}%` }}
            />
          </div>
          <span className="w-10 text-right text-xs tabular-nums text-slate-500">
            {Math.round(p * 100)}%
          </span>
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 4: Implement Spinner and ErrorNote**

`app/frontend/src/components/Spinner.tsx`:
```tsx
import { Loader2 } from "lucide-react";

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-slate-500">
      <Loader2 className="h-4 w-4 animate-spin" />
      {label && <span className="text-sm">{label}</span>}
    </div>
  );
}
```

`app/frontend/src/components/ErrorNote.tsx`:
```tsx
import { AlertTriangle } from "lucide-react";

export function ErrorNote({ message }: { message: string }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
      <AlertTriangle className="h-4 w-4" />
      <span>{message}</span>
    </div>
  );
}
```

- [ ] **Step 5: Implement providers and layout**

`app/frontend/src/app/providers.tsx`:
```tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient());
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
```

`app/frontend/src/app/layout.tsx`:
```tsx
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd app/frontend && npm run test -- proba-bars`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/app/providers.tsx app/frontend/src/app/layout.tsx app/frontend/src/components/Spinner.tsx app/frontend/src/components/ErrorNote.tsx app/frontend/src/components/ProbaBars.tsx app/frontend/tests/proba-bars.test.tsx
git commit -m "feat(frontend): root layout, query provider, shared primitives"
```

---

### Task 3: VerdictCard + ExplainPanel

**Files:**
- Create: `app/frontend/src/components/VerdictCard.tsx`, `app/frontend/src/components/ExplainPanel.tsx`
- Test: `app/frontend/tests/verdict-card.test.tsx`, `app/frontend/tests/explain-panel.test.tsx`

**Interfaces:**
- Consumes: `PredictResponse`, `TokenScore` from `lib/api`; `labelColor`, `labelVi`, `LABEL_NAMES` from `lib/labels`; `<ProbaBars/>`.
- Produces: `<VerdictCard result={PredictResponse}/>` — big label name + Vietnamese gloss + model name + `<ProbaBars/>`. `<ExplainPanel tokens={TokenScore[]}/>` — renders each token as a chip; positive score → red-tinted background, negative → green-tinted, zero → neutral; intensity scales with |score| relative to the max |score|.

- [ ] **Step 1: Write the failing tests**

`app/frontend/tests/verdict-card.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { VerdictCard } from "@/components/VerdictCard";

describe("VerdictCard", () => {
  it("shows label name, Vietnamese gloss and model", () => {
    render(<VerdictCard result={{ label: 2, label_name: "HATE", proba: [0.1, 0.2, 0.7], tokens: [], model: "PhoBERT-base-v2" }} />);
    expect(screen.getByText("HATE")).toBeInTheDocument();
    expect(screen.getByText("Thù ghét")).toBeInTheDocument();
    expect(screen.getByText(/PhoBERT-base-v2/)).toBeInTheDocument();
  });
});
```

`app/frontend/tests/explain-panel.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ExplainPanel } from "@/components/ExplainPanel";

describe("ExplainPanel", () => {
  it("renders a chip per token", () => {
    render(<ExplainPanel tokens={[{ token: "đồ", score: 0.0 }, { token: "ngu", score: 1.5 }]} />);
    expect(screen.getByText("đồ")).toBeInTheDocument();
    expect(screen.getByText("ngu")).toBeInTheDocument();
  });
  it("shows a hint when there are no tokens", () => {
    render(<ExplainPanel tokens={[]} />);
    expect(screen.getByText(/không có/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd app/frontend && npm run test -- verdict-card explain-panel`
Expected: FAIL — components not found.

- [ ] **Step 3: Implement VerdictCard**

`app/frontend/src/components/VerdictCard.tsx`:
```tsx
import type { PredictResponse } from "@/lib/api";
import { labelColor, labelVi } from "@/lib/labels";
import { ProbaBars } from "./ProbaBars";

export function VerdictCard({ result }: { result: PredictResponse }) {
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-baseline justify-between">
        <div>
          <div className={`text-3xl font-bold ${labelColor(result.label)}`}>{result.label_name}</div>
          <div className="text-sm text-slate-500">{labelVi(result.label)}</div>
        </div>
        <div className="text-xs text-slate-400">Mô hình: {result.model}</div>
      </div>
      <ProbaBars proba={result.proba} />
    </div>
  );
}
```

- [ ] **Step 4: Implement ExplainPanel**

`app/frontend/src/components/ExplainPanel.tsx`:
```tsx
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
```

Note: the dynamic `bg-red-${level}` classes must survive Tailwind's content scan. Add this safelist to `tailwind.config.ts` `theme`-level config:
```ts
// in tailwind.config.ts, add at top level of the config object:
safelist: [
  "bg-red-50","bg-red-100","bg-red-200","bg-emerald-50","bg-emerald-100","bg-emerald-200",
  "text-red-800","text-emerald-800",
],
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd app/frontend && npm run test -- verdict-card explain-panel`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/components/VerdictCard.tsx app/frontend/src/components/ExplainPanel.tsx app/frontend/tailwind.config.ts app/frontend/tests/verdict-card.test.tsx app/frontend/tests/explain-panel.test.tsx
git commit -m "feat(frontend): verdict card + token explanation panel"
```

---

### Task 4: ShowdownTable + RewriteCard

**Files:**
- Create: `app/frontend/src/components/ShowdownTable.tsx`, `app/frontend/src/components/RewriteCard.tsx`
- Test: `app/frontend/tests/showdown-table.test.tsx`

**Interfaces:**
- Consumes: `ShowdownResponse`, `ModelResult`, `RewriteResponse` from `lib/api`; `labelColor`, `labelVi` from `lib/labels`.
- Produces: `<ShowdownTable models={ModelResult[]}/>` — one row per model: display_name, predicted label (colored), top-class confidence %, latency ms; rows tagged `data-testid="showdown-row"`; a consensus badge showing whether all models agree. `<RewriteCard data={RewriteResponse}/>` — shows rewritten text, before→after labels with an arrow, animated with framer-motion.

- [ ] **Step 1: Write the failing test**

`app/frontend/tests/showdown-table.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ShowdownTable } from "@/components/ShowdownTable";

const models = [
  { name: "PhoBERT", display_name: "PhoBERT-base-v2", label: 2, proba: [0.1, 0.1, 0.8], latency_ms: 30 },
  { name: "LogisticRegression", display_name: "Logistic Regression", label: 2, proba: [0.2, 0.2, 0.6], latency_ms: 2 },
];

describe("ShowdownTable", () => {
  it("renders one row per model with confidence", () => {
    render(<ShowdownTable models={models} />);
    expect(screen.getAllByTestId("showdown-row")).toHaveLength(2);
    expect(screen.getByText("PhoBERT-base-v2")).toBeInTheDocument();
    expect(screen.getByText("80%")).toBeInTheDocument();
  });
  it("shows consensus when all agree", () => {
    render(<ShowdownTable models={models} />);
    expect(screen.getByText(/đồng thuận/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npm run test -- showdown-table`
Expected: FAIL — component not found.

- [ ] **Step 3: Implement ShowdownTable**

`app/frontend/src/components/ShowdownTable.tsx`:
```tsx
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
```

- [ ] **Step 4: Implement RewriteCard**

`app/frontend/src/components/RewriteCard.tsx`:
```tsx
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd app/frontend && npm run test -- showdown-table`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/components/ShowdownTable.tsx app/frontend/src/components/RewriteCard.tsx app/frontend/tests/showdown-table.test.tsx
git commit -m "feat(frontend): showdown table + rewrite card"
```

---

### Task 5: Studio page (`/`) — InputBar + pipeline wiring

**Files:**
- Create: `app/frontend/src/components/InputBar.tsx`, `app/frontend/src/app/page.tsx`
- Test: none new (page wiring is integration; covered by component tests + manual run). Run the full suite + a typecheck instead.

**Interfaces:**
- Consumes: `predict`, `showdown`, `rewrite` from `lib/api`; all components from Tasks 2–4; `useMutation` from React Query.
- Produces: the Studio route. `<InputBar onAnalyze={(text)=>void} loading={boolean}/>` with a textarea, sample chips (one CLEAN, one OFFENSIVE, one HATE example), and an "Phân tích" button.

- [ ] **Step 1: Implement InputBar**

`app/frontend/src/components/InputBar.tsx`:
```tsx
"use client";
import { useState } from "react";

const SAMPLES = [
  "Cảm ơn bạn nhiều nhé, bài viết rất hữu ích",
  "Nói chuyện vô duyên thế, phiền phức quá",
  "Lũ người đó đáng bị tiêu diệt hết",
];

export function InputBar({ onAnalyze, loading }: { onAnalyze: (t: string) => void; loading: boolean }) {
  const [text, setText] = useState("");
  return (
    <div className="rounded-2xl border bg-white p-4 shadow-sm">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        placeholder="Nhập bình luận tiếng Việt..."
        className="w-full resize-none rounded-lg border border-slate-200 p-3 text-slate-800 outline-none focus:border-slate-400"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {SAMPLES.map((s, i) => (
          <button key={i} onClick={() => setText(s)}
            className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600 hover:bg-slate-200">
            Ví dụ {i + 1}
          </button>
        ))}
        <button
          onClick={() => onAnalyze(text)}
          disabled={loading || text.trim().length === 0}
          className="ml-auto rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {loading ? "Đang phân tích..." : "Phân tích"}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Implement the Studio page**

`app/frontend/src/app/page.tsx`:
```tsx
"use client";
import { useMutation } from "@tanstack/react-query";
import { predict, showdown, rewrite } from "@/lib/api";
import { InputBar } from "@/components/InputBar";
import { VerdictCard } from "@/components/VerdictCard";
import { ExplainPanel } from "@/components/ExplainPanel";
import { ShowdownTable } from "@/components/ShowdownTable";
import { RewriteCard } from "@/components/RewriteCard";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";
import { useState } from "react";

export default function StudioPage() {
  const [current, setCurrent] = useState("");
  const predictM = useMutation({ mutationFn: predict });
  const showdownM = useMutation({ mutationFn: showdown });
  const rewriteM = useMutation({ mutationFn: rewrite });

  function analyze(text: string) {
    setCurrent(text);
    rewriteM.reset();
    predictM.mutate(text);
    showdownM.mutate(text);
  }

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-slate-900">Comment Moderation Studio</h1>
      <InputBar onAnalyze={analyze} loading={predictM.isPending} />

      {predictM.isError && <ErrorNote message={(predictM.error as Error).message} />}
      {predictM.isPending && <Spinner label="Đang chấm điểm..." />}
      {predictM.data && (
        <>
          <VerdictCard result={predictM.data} />
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <h3 className="mb-3 font-semibold text-slate-800">Vì sao? (giải thích theo mô hình tuyến tính)</h3>
            <ExplainPanel tokens={predictM.data.tokens} />
          </div>
        </>
      )}

      {showdownM.isPending && <Spinner label="Đang chạy 7 mô hình..." />}
      {showdownM.data && <ShowdownTable models={showdownM.data.models} />}

      {predictM.data && (
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <button
            onClick={() => rewriteM.mutate(current)}
            disabled={rewriteM.isPending}
            className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {rewriteM.isPending ? "Đang viết lại..." : "Viết lại lịch sự"}
          </button>
          {rewriteM.isError && <div className="mt-3"><ErrorNote message={(rewriteM.error as Error).message} /></div>}
          {rewriteM.data && <div className="mt-4"><RewriteCard data={rewriteM.data} /></div>}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Typecheck + full suite**

Run: `cd app/frontend && npx tsc --noEmit && npm run test`
Expected: tsc clean; all component/lib tests pass.

- [ ] **Step 4: Commit**

```bash
git add app/frontend/src/components/InputBar.tsx app/frontend/src/app/page.tsx
git commit -m "feat(frontend): Studio page wiring predict/showdown/rewrite"
```

---

### Task 6: Simulate page (`/simulate`) — Dropzone + BatchDashboard

**Files:**
- Create: `app/frontend/src/components/simulate/Dropzone.tsx`, `app/frontend/src/components/simulate/BatchDashboard.tsx`, `app/frontend/src/app/simulate/page.tsx`
- Test: `app/frontend/tests/batch-dashboard.test.tsx`

**Interfaces:**
- Consumes: `batch`, `BatchResponse` from `lib/api`; `recharts` PieChart; `labelColor` from `lib/labels`.
- Produces: `<Dropzone onFile={(file:File)=>void} disabled/>` (file input accepting `.csv`); `<BatchDashboard data={BatchResponse}/>` — total count, toxic ratio %, a donut of the 3-class counts, and a table of the most-toxic rows (sorted by `proba[2]` desc, top 20).

- [ ] **Step 1: Write the failing test**

`app/frontend/tests/batch-dashboard.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { BatchDashboard } from "@/components/simulate/BatchDashboard";

const data = {
  total: 3,
  counts: { CLEAN: 1, OFFENSIVE: 1, HATE: 1 },
  toxic_ratio: 0.6667,
  rows: [
    { text: "a", label: 2, label_name: "HATE", proba: [0.1, 0.2, 0.7] },
    { text: "b", label: 0, label_name: "CLEAN", proba: [0.9, 0.05, 0.05] },
    { text: "c", label: 1, label_name: "OFFENSIVE", proba: [0.2, 0.7, 0.1] },
  ],
};

describe("BatchDashboard", () => {
  it("shows total and toxic ratio", () => {
    render(<BatchDashboard data={data} />);
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText(/67%/)).toBeInTheDocument();
  });
  it("lists the most toxic row first", () => {
    render(<BatchDashboard data={data} />);
    const rows = screen.getAllByTestId("toxic-row");
    expect(rows[0]).toHaveTextContent("a");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npm run test -- batch-dashboard`
Expected: FAIL — component not found.

- [ ] **Step 3: Implement Dropzone**

`app/frontend/src/components/simulate/Dropzone.tsx`:
```tsx
"use client";
import { UploadCloud } from "lucide-react";

export function Dropzone({ onFile, disabled }: { onFile: (f: File) => void; disabled?: boolean }) {
  return (
    <label className="flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 bg-white p-8 text-slate-500 hover:border-slate-400">
      <UploadCloud className="h-8 w-8" />
      <span className="text-sm">Tải lên file CSV (cột <code>free_text</code>)</span>
      <input
        type="file"
        accept=".csv"
        disabled={disabled}
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
      />
    </label>
  );
}
```

- [ ] **Step 4: Implement BatchDashboard**

`app/frontend/src/components/simulate/BatchDashboard.tsx`:
```tsx
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
```

- [ ] **Step 5: Implement the simulate page**

`app/frontend/src/app/simulate/page.tsx`:
```tsx
"use client";
import { useMutation } from "@tanstack/react-query";
import { batch } from "@/lib/api";
import { Dropzone } from "@/components/simulate/Dropzone";
import { BatchDashboard } from "@/components/simulate/BatchDashboard";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";

export default function SimulatePage() {
  const m = useMutation({ mutationFn: batch });
  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-slate-900">Mô phỏng kiểm duyệt hàng loạt</h1>
      <Dropzone onFile={(f) => m.mutate(f)} disabled={m.isPending} />
      {m.isPending && <Spinner label="Đang chấm điểm file..." />}
      {m.isError && <ErrorNote message={(m.error as Error).message} />}
      {m.data && <BatchDashboard data={m.data} />}
    </div>
  );
}
```

- [ ] **Step 6: Run tests + typecheck**

Run: `cd app/frontend && npm run test -- batch-dashboard && npx tsc --noEmit`
Expected: PASS; tsc clean.

- [ ] **Step 7: Commit**

```bash
git add app/frontend/src/components/simulate app/frontend/src/app/simulate/page.tsx app/frontend/tests/batch-dashboard.test.tsx
git commit -m "feat(frontend): simulate page with CSV dropzone + batch dashboard"
```

---

### Task 7: Insights page (`/insights`) — MetricsTable

**Files:**
- Create: `app/frontend/src/components/insights/MetricsTable.tsx`, `app/frontend/src/app/insights/page.tsx`
- Test: `app/frontend/tests/metrics-table.test.tsx`

**Interfaces:**
- Consumes: `getInsights`, `InsightsResponse`, `ModelMetric` from `lib/api`; `useQuery` from React Query.
- Produces: `<MetricsTable data={InsightsResponse}/>` — a sortable-looking table of the 7 models with accuracy/precision_w/recall_w/f1_w/f1_macro (formatted to 4 decimals), the `best` model row highlighted. The insights page fetches via `useQuery` and renders loading/error states.

- [ ] **Step 1: Write the failing test**

`app/frontend/tests/metrics-table.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MetricsTable } from "@/components/insights/MetricsTable";

const data = {
  best: "PhoBERT-base-v2",
  models: [
    { display_name: "PhoBERT-base-v2", accuracy: 0.8558, precision_w: 0.8733, recall_w: 0.8558, f1_w: 0.8637, f1_macro: 0.6703 },
    { display_name: "Random Forest", accuracy: 0.8290, precision_w: 0.7975, recall_w: 0.8290, f1_w: 0.8080, f1_macro: 0.5101 },
  ],
};

describe("MetricsTable", () => {
  it("renders a row per model with formatted metrics", () => {
    render(<MetricsTable data={data} />);
    expect(screen.getAllByTestId("metric-row")).toHaveLength(2);
    expect(screen.getByText("0.6703")).toBeInTheDocument();
  });
  it("highlights the best model", () => {
    render(<MetricsTable data={data} />);
    const best = screen.getByTestId("metric-row-best");
    expect(best).toHaveTextContent("PhoBERT-base-v2");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd app/frontend && npm run test -- metrics-table`
Expected: FAIL — component not found.

- [ ] **Step 3: Implement MetricsTable**

`app/frontend/src/components/insights/MetricsTable.tsx`:
```tsx
import type { InsightsResponse } from "@/lib/api";

const COLS: { key: keyof Omit<InsightsResponse["models"][number], "display_name">; label: string }[] = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision_w", label: "Precision_w" },
  { key: "recall_w", label: "Recall_w" },
  { key: "f1_w", label: "F1_w" },
  { key: "f1_macro", label: "F1_macro" },
];

export function MetricsTable({ data }: { data: InsightsResponse }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-left text-xs uppercase text-slate-400">
          <th className="pb-2">Mô hình</th>
          {COLS.map((c) => <th key={c.key} className="pb-2">{c.label}</th>)}
        </tr>
      </thead>
      <tbody>
        {data.models.map((m) => {
          const isBest = m.display_name === data.best;
          return (
            <tr
              key={m.display_name}
              data-testid={isBest ? "metric-row-best metric-row" : "metric-row"}
              className={`border-t ${isBest ? "bg-emerald-50 font-semibold" : ""}`}
            >
              <td className="py-2 text-slate-700">{m.display_name}</td>
              {COLS.map((c) => (
                <td key={c.key} className="py-2 tabular-nums text-slate-600">{m[c.key].toFixed(4)}</td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
```

Note: `getAllByTestId("metric-row")` matches both the plain and the best row because the best row's `data-testid` includes `metric-row` as a space-separated token. (Testing Library matches the full attribute string; if your version matches exact, set the best row testid to exactly `metric-row` and add a separate `data-best="true"` attribute and assert on it instead.)

- [ ] **Step 4: Implement the insights page**

`app/frontend/src/app/insights/page.tsx`:
```tsx
"use client";
import { useQuery } from "@tanstack/react-query";
import { getInsights } from "@/lib/api";
import { MetricsTable } from "@/components/insights/MetricsTable";
import { Spinner } from "@/components/Spinner";
import { ErrorNote } from "@/components/ErrorNote";

export default function InsightsPage() {
  const { data, isPending, isError, error } = useQuery({
    queryKey: ["insights"],
    queryFn: getInsights,
  });
  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-slate-900">Kết quả mô hình (test set)</h1>
      {isPending && <Spinner label="Đang tải..." />}
      {isError && <ErrorNote message={(error as Error).message} />}
      {data && (
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          <MetricsTable data={data} />
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Run tests + typecheck**

Run: `cd app/frontend && npm run test -- metrics-table && npx tsc --noEmit`
Expected: PASS; tsc clean.

- [ ] **Step 6: Commit**

```bash
git add app/frontend/src/components/insights app/frontend/src/app/insights/page.tsx app/frontend/tests/metrics-table.test.tsx
git commit -m "feat(frontend): insights page with model metrics table"
```

---

### Task 8: Build verification + README + full suite

**Files:**
- Create: `app/frontend/README.md`
- Modify: repo `.gitignore` (ensure `app/frontend/node_modules/`, `app/frontend/.next/` ignored)
- Test: full suite + production build

**Interfaces:** none new.

- [ ] **Step 1: Ensure ignores**

Confirm the repo `.gitignore` contains:
```
app/frontend/node_modules/
app/frontend/.next/
app/frontend/.env.local
```
Add any missing lines.

- [ ] **Step 2: Write README**

`app/frontend/README.md`:
```markdown
# ViHSD Frontend (Next.js)

Comment Moderation Studio UI. Consumes the FastAPI backend.

## Setup
```
cd app/frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL to your backend
npm run dev                        # http://localhost:3000
```

## Routes
- `/` — Studio: predict + explain + showdown + rewrite
- `/simulate` — batch CSV scoring dashboard
- `/insights` — model metrics table

## Test
```
npm run test       # vitest component/lib tests
npm run build      # production build check
```

Backend must be running at `NEXT_PUBLIC_API_URL` for live data.
```

- [ ] **Step 3: Run the full test suite**

Run: `cd app/frontend && npm run test`
Expected: all suites pass (labels, api, proba-bars, verdict-card, explain-panel, showdown-table, batch-dashboard, metrics-table).

- [ ] **Step 4: Verify production build**

Run: `cd app/frontend && npm run build`
Expected: build completes with all 3 routes (`/`, `/simulate`, `/insights`) compiled, no type errors.

- [ ] **Step 5: Commit**

```bash
git add app/frontend/README.md .gitignore
git commit -m "docs(frontend): README + ignore build artifacts; verify build"
```

---

## Self-Review

**Spec coverage (design §5):**
- §5 routes `/`, `/simulate`, `/insights` → Tasks 5, 6, 7. ✅
- §5.1 Studio: InputBar+samples (T5), VerdictCard+gauge/proba (T3/T2), ExplainPanel highlight (T3), ShowdownTable (T4), RewriteCard before→after+animation (T4), lazy calls + skeleton/loading + error (T5 wiring with Spinner/ErrorNote). ✅ (gauge rendered as ProbaBars + big label; a literal radial gauge is optional polish.)
- §5.2 simulate: dropzone CSV → /batch → donut + toxic ratio + top toxic table (T6). ✅
- §5.3 insights: metrics table, best highlighted (T7). Figure grid of report PNGs is deferred — see note. ⚠️
- Global: NEXT_PUBLIC_API_URL, label colors, Vietnamese copy, loading/error everywhere → Tasks 1,2,5,6,7. ✅
- Backend contract types match exactly (§ Global Constraints types vs lib/api.ts). ✅

**Deferred / deviations (intentional):**
- shadcn CLI → hand-rolled Tailwind components (stated in Global Constraints).
- §5.3 "figure grid" of report PNGs (confusion matrices etc.) is NOT included — the metrics table covers the numeric story and avoids bundling/serving the LaTeX figure assets into the Next app. Add later if desired by copying PNGs into `public/` and rendering an image grid.
- Literal radial confidence gauge replaced by labelled proba bars + prominent colored verdict (functionally equivalent; upgrade during visual polish).

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✅

**Type consistency:** `predict/showdown/rewrite/batch/getInsights` signatures and the `Verdict/PredictResponse/ModelResult/ShowdownResponse/RewriteResponse/BatchResponse/InsightsResponse` types are defined in Task 1 and used unchanged in Tasks 3–7. `labelColor/labelBg/labelVi/LABEL_NAMES` defined in Task 1, used consistently. Component prop names (`result`, `tokens`, `models`, `data`, `proba`) are stable across tasks. ✅

---

## Execution Handoff

This is **Plan 2 of 3**. Plan 3 (Deploy: backend→HF Space, frontend→Vercel, CORS wiring) follows. Recommended execution: subagent-driven (fresh subagent per task, review between).
