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
