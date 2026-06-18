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
    // label cell shows the Vietnamese gloss, not the raw integer
    expect(screen.getAllByText("Thù ghét").length).toBeGreaterThan(0);
  });
  it("shows consensus when all agree", () => {
    render(<ShowdownTable models={models} />);
    expect(screen.getByText(/đồng thuận/i)).toBeInTheDocument();
  });
});
