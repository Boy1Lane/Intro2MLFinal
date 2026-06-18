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
