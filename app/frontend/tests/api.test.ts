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
