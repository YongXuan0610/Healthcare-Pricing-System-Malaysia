import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import AIPredictionForm, {
  buildNSSPayload,
  NSS_FIELD_GROUPS,
  type NSSFormData,
  type NSSOptions,
} from "@/components/AIPredictionForm";

vi.mock("@/lib/supabase", () => ({
  supabase: { auth: { getSession: vi.fn(async () => ({ data: { session: null }, error: null })) } },
}));

const validForm: NSSFormData = {
  age_years: "35",
  gender: "Female",
  chronic_ailment: "No",
  pregnant: "No",
  communicable_disease: "Not suffered",
  other_ailment_last_15_days: "No",
  number_of_hospitalisations: "1",
  length_of_stay_days: "3",
  ailment_nature: "All other fevers",
  hospitalisation_treatment_nature: "Allopathy",
  medical_institution_type: "Private hospital",
  ward_type: "Paying general ward",
  place_of_hospitalisation: "Same district, urban area",
  surgery: "Not received",
  medicine: "Received",
};

const optionalFields: (keyof NSSFormData)[] = ["pregnant"];
const fieldKeys = NSS_FIELD_GROUPS.flatMap((group) => group.fields.map((field) => field.key));
const numericKeys = new Set(
  NSS_FIELD_GROUPS.flatMap((group) => group.fields)
    .filter((field) => field.type === "number")
    .map((field) => field.key),
);

const options: NSSOptions = {
  field_order: fieldKeys,
  optional_fields: optionalFields,
  numeric_limits: Object.fromEntries(
    [...numericKeys].map((key) => [key, { minimum: 0, maximum: 999999, step: 1 }]),
  ),
  categories: Object.fromEntries(
    fieldKeys.filter((key) => !numericKeys.has(key)).map((key) => [key, [validForm[key]]]),
  ),
  model: {
    model_version: "nss80_practical_15",
    model_type: "HistGradientBoostingRegressor",
    architecture: "Global model",
    target_transform: "identity",
    feature_count: 15,
    encoded_feature_count: 90,
    r2_test: 0.39769617865686857,
    mae_test_inr: 19614.1828531067,
    rmse_test_inr: 57639.143726165756,
    median_absolute_error_test_inr: 5436.29534237078,
    rmsle_test: 3.3737125327591118,
  },
  prediction_interval: { available: true, reason: "Person-calibrated 80% interval is available." },
};

const jsonResponse = (data: unknown): Response =>
  ({ ok: true, json: async () => data }) as Response;

describe("AIPredictionForm NSS practical 15", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).includes("/predict/nss80/options")) return jsonResponse(options);
        if (String(input).includes("/predict/nss80") && init?.method === "POST") {
          return jsonResponse({
            predicted_medical_expenditure: 26275.74,
            predicted_medical_expenditure_range: { lower: 0, upper: 80000, coverage_level: 0.8, currency: "INR" },
            prediction_interval_available: true,
            currency: "INR",
            limitation: "Research estimate; not a Malaysian price.",
            range_interpretation: "80% person-calibrated range; not an individual guarantee.",
            model_version: "nss80_practical_15",
            model_type: "HistGradientBoostingRegressor",
            r2_test: 0.39769617865686857,
          });
        }
        throw new Error(`Unexpected request: ${String(input)}`);
      }),
    );
  });

  it("defines the exact 15-field contract in the required two sections", () => {
    expect(NSS_FIELD_GROUPS.map((group) => group.title)).toEqual([
      "Patient and Health Context",
      "Hospitalisation Episode",
    ]);
    expect(NSS_FIELD_GROUPS.map((group) => group.fields.length)).toEqual([6, 9]);
    expect(fieldKeys).toEqual([
      "age_years", "gender", "chronic_ailment", "pregnant", "communicable_disease",
      "other_ailment_last_15_days", "number_of_hospitalisations", "length_of_stay_days",
      "ailment_nature", "hospitalisation_treatment_nature", "medical_institution_type",
      "ward_type", "place_of_hospitalisation", "surgery", "medicine",
    ]);

    const payload = buildNSSPayload(validForm, optionalFields);
    expect(Object.keys(payload)).toEqual(fieldKeys);
    expect(payload.age_years).toBe(35);
    expect(payload.number_of_hospitalisations).toBe(1);
    expect(payload.medicine).toBe("Received");
  });

  it("renders a responsive two-column form and submits only practical inputs", async () => {
    render(<AIPredictionForm />);
    const submit = await screen.findByRole("button", { name: "Generate NSS Primary Estimate" });
    expect(screen.getByRole("heading", { name: "NSS 80 Healthcare Expenditure Model" })).toBeInTheDocument();
    expect(screen.getByText("Uses 15 leakage-safe patient and hospitalisation inputs to estimate medical expenditure for an inpatient case in India. Results are shown in INR.")).toBeInTheDocument();
    expect(screen.queryByText(/Locked-test metrics/)).not.toBeInTheDocument();
    expect(screen.queryByText(/encoded/)).not.toBeInTheDocument();
    expect(screen.queryByText(/R² is not classification accuracy/)).not.toBeInTheDocument();
    const patient = screen.getByRole("heading", { name: "Patient and Health Context" }).closest("section")!;
    const episode = screen.getByRole("heading", { name: "Hospitalisation Episode" }).closest("section")!;
    expect(patient.querySelector(".sm\\:grid-cols-2")).toBeInTheDocument();
    expect(episode.querySelector(".sm\\:grid-cols-2")).toBeInTheDocument();
    expect(within(patient).getAllByLabelText(/./)).toHaveLength(6);
    expect(within(episode).getAllByLabelText(/./)).toHaveLength(9);
    expect(document.querySelectorAll('[id^="nss-"]')).toHaveLength(15);
    expect(screen.queryByRole("heading", { name: /Household and location/i })).not.toBeInTheDocument();
    for (const obsoleteLabel of [
      /Relationship to household head/i,
      /Marital status/i,
      /Highest education/i,
      /Health financing/i,
      /insurance premium/i,
      /Other ailment previous day/i,
      /Reason for not using/i,
      /Treatment state/i,
      /X-ray|scan received/i,
      /Other diagnostic tests/i,
    ]) {
      expect(screen.queryByLabelText(obsoleteLabel)).not.toBeInTheDocument();
    }

    for (const [key, value] of Object.entries(validForm)) {
      fireEvent.change(document.getElementById(`nss-${key}`) as HTMLElement, { target: { value } });
    }
    fireEvent.click(submit);

    const primaryRange = await screen.findByText("INR 0.00 – INR 80,000.00");
    expect(primaryRange).toHaveClass("text-3xl");
    expect(screen.getByText("Estimated Medical Expenditure Range")).toBeInTheDocument();
    expect(screen.getByText("Estimated 80% prediction range.")).toBeInTheDocument();
    expect(screen.getByText("Central model estimate: INR 26,275.74")).not.toHaveClass("text-3xl");
    expect(screen.getByText("Actual expenditure may vary. Based on India’s NSS 80 dataset; not a Malaysian hospital quotation.")).toBeInTheDocument();
    expect(screen.queryByText(/similar hospitalisation records/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Primary research estimate/)).not.toBeInTheDocument();
    expect(screen.queryByText(/person-calibrated/)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Prediction range visualization")).toHaveClass("max-w-xl");
    expect(screen.getByLabelText("Central model estimate marker")).toBeInTheDocument();
    await waitFor(() => {
      const postCall = vi.mocked(fetch).mock.calls.find(([, init]) => init?.method === "POST");
      const payload = JSON.parse(String(postCall?.[1]?.body));
      expect(Object.keys(payload)).toEqual(fieldKeys);
      expect(payload).not.toHaveProperty("treatment_state_code");
      expect(payload).not.toHaveProperty("health_financing_or_insurance_coverage");
      expect(payload).not.toHaveProperty("xray_ecg_eeg_scan");
    });
  });

  it("applies pregnancy applicability and required-field validation", async () => {
    render(<AIPredictionForm />);
    await screen.findByRole("button", { name: "Generate NSS Primary Estimate" });
    const pregnancy = screen.getByLabelText("Pregnant");
    expect(pregnancy).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Age (years)"), { target: { value: "35" } });
    fireEvent.change(screen.getByLabelText("Gender recorded by NSS"), { target: { value: "Female" } });
    expect(pregnancy).toBeEnabled();
    expect(pregnancy).toBeRequired();
    expect(screen.getByLabelText("Nature of ailment")).toBeRequired();
  });

  it("shows a controlled error and blocks submission when options fail", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: false,
      json: async () => ({ detail: "NSS practical 15 is unavailable." }),
    }) as Response));
    render(<AIPredictionForm />);
    expect(await screen.findByText(
      "The NSS practical 15 codebook options could not be loaded from the API.",
    )).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate NSS Primary Estimate" })).toBeDisabled();
  });
});
