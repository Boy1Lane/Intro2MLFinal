import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  safelist: [
    "bg-red-50","bg-red-100","bg-red-200","bg-emerald-50","bg-emerald-100","bg-emerald-200",
    "text-red-800","text-emerald-800",
  ],
  theme: { extend: {} },
  plugins: [],
};
export default config;
