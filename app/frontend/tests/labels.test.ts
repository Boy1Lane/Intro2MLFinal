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
