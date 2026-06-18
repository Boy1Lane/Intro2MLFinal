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
    const { container } = render(<MetricsTable data={data} />);
    const best = container.querySelector('[data-best="true"]');
    expect(best).toBeInTheDocument();
    expect(best).toHaveTextContent("PhoBERT-base-v2");
  });
});
