import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { VerdictCard } from "@/components/VerdictCard";

describe("VerdictCard", () => {
  it("shows label name, Vietnamese gloss and model", () => {
    render(<VerdictCard result={{ label: 2, label_name: "HATE", proba: [0.1, 0.2, 0.7], tokens: [], model: "PhoBERT-base-v2" }} />);
    expect(screen.getAllByText("HATE")).toHaveLength(2);
    expect(screen.getByText("Thù ghét")).toBeInTheDocument();
    expect(screen.getByText(/PhoBERT-base-v2/)).toBeInTheDocument();
  });
});
