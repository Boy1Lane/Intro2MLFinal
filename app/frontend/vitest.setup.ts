import "@testing-library/jest-dom/vitest";

// Polyfill ResizeObserver for recharts in jsdom
if (typeof global.ResizeObserver === "undefined") {
  global.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
