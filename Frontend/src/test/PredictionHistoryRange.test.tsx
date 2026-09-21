import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import PredictionHistory from "@/pages/PredictionHistory";

const rows = [
  {
    id: "old-point-only",
    user_id: "user-1",
    hospital_type: "public",
    total_cost: 120,
    currency: "MYR",
    inputs: { category: "ward charges" },
    breakdown: [],
    created_at: "2026-09-10T00:00:00Z",
  },
  {
    id: "new-nss-range",
    user_id: "user-1",
    hospital_type: "nss80",
    total_cost: 5432.31,
    currency: "INR",
    inputs: { age_years: 35 },
    breakdown: { lower_bound: 0, upper_bound: 11400.21, interval_level: 0.8 },
    created_at: "2026-09-10T01:00:00Z",
  },
];

vi.mock("@/lib/supabase", () => {
  const query = {
    select: vi.fn(),
    eq: vi.fn(),
    order: vi.fn(async () => ({ data: rows, error: null })),
    delete: vi.fn(),
  };
  query.select.mockReturnValue(query);
  query.eq.mockReturnValue(query);
  query.delete.mockReturnValue(query);
  return {
    supabase: {
      auth: { getUser: vi.fn(async () => ({ data: { user: { id: "user-1" } }, error: null })) },
      from: vi.fn(() => query),
    },
  };
});

describe("PredictionHistory range compatibility", () => {
  it("renders an old point-only record and a new NSS interval record safely", async () => {
    render(
      <MemoryRouter>
        <PredictionHistory />
      </MemoryRouter>,
    );

    expect(await screen.findByText("NSS 80 Research Estimate")).toBeInTheDocument();
    expect(screen.getByText("INR 0.00 – INR 11,400.21")).toBeInTheDocument();
    expect(screen.getByText(/Estimated 80% prediction range/)).toBeInTheDocument();
    expect(screen.getByText(/Central model estimate: INR 5,432.31/)).toBeInTheDocument();
    expect(screen.getByText("RM 120.00")).toBeInTheDocument();
    expect(screen.getByText(/No detailed cost breakdown was saved/)).toBeInTheDocument();
  });
});
