import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import LIAMPricingReference from "@/components/LIAMPricingReference";


const sourceUrl =
  "https://www.liam.org.my/library/healthcare/Price-Ranges-Common-Private-Healthcare-Services-Msia_27Jan2026.pdf";

const procedures = {
  procedures: [
    {
      procedure_code: "M6",
      procedure_name: "Treatment for Wrist Fracture",
      body_system: "Musculoskeletal and Connective Tissue",
      care_setting: "Inpatient",
      facility_states: [
        { state: "Johor", data_status: "Published" },
        { state: "Perlis", data_status: "Insufficient credible data" },
        { state: "Selangor", data_status: "Published" },
      ],
    },
    {
      procedure_code: "C1",
      procedure_name: "Angiogram",
      body_system: "Circulatory",
      care_setting: "Inpatient",
      facility_states: [{ state: "Johor", data_status: "Published" }],
    },
  ],
};

const publishedResult = (
  segment: string,
  lower: number,
  typical: number,
  upper: number,
  page: number,
  discharges: string,
) => ({
  estimate_available: true,
  pricing_type: "published_percentile_range",
  currency: "MYR",
  procedure_name: "Treatment for Wrist Fracture",
  care_setting: "Inpatient",
  segmentation_type: segment === "All" ? "Overall" : "Facility State",
  segment,
  data_status: "Published",
  lower_reference: lower,
  typical_bill_amount: typical,
  upper_reference: upper,
  number_of_discharges_band: discharges,
  source_pdf_page: page,
  source_url: sourceUrl,
  message: "Published LIAM reference only.",
});

const jsonResponse = (data: unknown): Response =>
  ({
    ok: true,
    json: async () => data,
  }) as Response;

describe("LIAMPricingReference", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const requestUrl = String(input);
        if (requestUrl.includes("/pricing/reference/liam/procedures")) {
          return jsonResponse(procedures);
        }

        const query = new URL(requestUrl).searchParams;
        const procedureCode = query.get("procedure_code");
        const segment = query.get("segment");

        if (procedureCode === "M6" && segment === "All") {
          return jsonResponse(
            publishedResult("All", 3900, 5800, 8900, 103, "1,000 - 4,999"),
          );
        }
        if (procedureCode === "M6" && segment === "Johor") {
          return jsonResponse(
            publishedResult("Johor", 4200, 6700, 9400, 104, "100 - 499"),
          );
        }
        if (procedureCode === "M6" && segment === "Selangor") {
          return jsonResponse(
            publishedResult("Selangor", 4000, 5900, 9300, 104, "500 - 999"),
          );
        }
        if (procedureCode === "M6" && segment === "Perlis") {
          return jsonResponse({
            estimate_available: false,
            pricing_type: "insufficient_credible_data",
            currency: "MYR",
            procedure_name: "Treatment for Wrist Fracture",
            care_setting: "Inpatient",
            segmentation_type: "Facility State",
            segment: "Perlis",
            data_status: "Insufficient credible data",
            source_pdf_page: 104,
            source_url: sourceUrl,
            message: "LIAM marks this segment as insufficient credible data.",
          });
        }
        if (procedureCode === "C1" && segment === "All") {
          return jsonResponse(
            publishedResult("All", 8800, 11700, 18300, 15, "5,000 - 9,999"),
          );
        }

        throw new Error(`Unexpected LIAM request: ${requestUrl}`);
      }),
    );
  });

  it("loads the overall reference by default and updates immediately by state", async () => {
    render(<LIAMPricingReference />);

    const procedureSelect = await screen.findByLabelText(
      "Procedure and care setting",
    );
    const stateSelect = screen.getByLabelText("Facility State");

    expect(stateSelect).toBeDisabled();
    fireEvent.change(procedureSelect, { target: { value: "M6::Inpatient" } });

    await screen.findByText("RM 3,900 – RM 8,900");
    expect(stateSelect).toHaveValue("__overall__");
    expect(screen.getByText("Malaysia – Overall")).toBeInTheDocument();
    expect(
      screen.getByRole("option", {
        name: "Perlis — Insufficient credible data",
      }),
    ).toBeInTheDocument();

    fireEvent.change(stateSelect, { target: { value: "Johor" } });
    await screen.findByText("RM 4,200 – RM 9,400");
    expect(screen.getByText("Reference sample: 100 - 499 discharges")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /LIAM source PDF, page 104/ })).toHaveAttribute(
      "href",
      `${sourceUrl}#page=104`,
    );

    fireEvent.change(stateSelect, { target: { value: "Selangor" } });
    await screen.findByText("RM 4,000 – RM 9,300");
    expect(screen.getByText("Published typical bill: RM 5,900")).toBeInTheDocument();
  });

  it("shows insufficient credible data without a zero or national fallback", async () => {
    render(<LIAMPricingReference />);

    fireEvent.change(await screen.findByLabelText("Procedure and care setting"), {
      target: { value: "M6::Inpatient" },
    });
    const stateSelect = screen.getByLabelText("Facility State");
    await waitFor(() => expect(stateSelect).toBeEnabled());
    fireEvent.change(stateSelect, { target: { value: "Perlis" } });

    await screen.findByText("Insufficient Credible Data");
    expect(screen.getByText(/available claims data were insufficient/)).toBeInTheDocument();
    expect(screen.queryByText(/RM 0/)).not.toBeInTheDocument();
    expect(screen.queryByText("RM 3,900 – RM 8,900")).not.toBeInTheDocument();
  });

  it("resets Facility State to overall when the procedure changes", async () => {
    render(<LIAMPricingReference />);

    const procedureSelect = await screen.findByLabelText(
      "Procedure and care setting",
    );
    const stateSelect = screen.getByLabelText("Facility State");
    fireEvent.change(procedureSelect, { target: { value: "M6::Inpatient" } });
    await screen.findByText("RM 3,900 – RM 8,900");
    fireEvent.change(stateSelect, { target: { value: "Johor" } });
    await screen.findByText("RM 4,200 – RM 9,400");

    fireEvent.change(procedureSelect, { target: { value: "C1::Inpatient" } });

    await screen.findByText("RM 8,800 – RM 18,300");
    expect(stateSelect).toHaveValue("__overall__");
    expect(screen.queryByRole("button", { name: /View LIAM Published Range/ })).not.toBeInTheDocument();
  });
});
