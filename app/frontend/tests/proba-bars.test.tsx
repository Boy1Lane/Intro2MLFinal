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
