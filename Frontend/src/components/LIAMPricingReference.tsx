import { useEffect, useState } from "react";
import { BookOpen, ExternalLink } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { apiUrl } from "@/lib/api";

type FacilityStateOption = {
  state: string;
  data_status: "Published" | "Insufficient credible data";
};

type Procedure = {
  procedure_code: string;
  procedure_name: string;
  body_system: string;
  care_setting: string;
  facility_states: FacilityStateOption[];
};

type LIAMResult = {
  estimate_available: boolean;
  pricing_type: string;
  currency: string;
  procedure_name?: string;
  care_setting?: string;
  segmentation_type?: string;
  segment?: string;
  data_status?: string;
  lower_reference?: number;
  typical_bill_amount?: number;
  upper_reference?: number;
  number_of_discharges_band?: string | null;
  source_pdf_page?: number;
  source_url?: string;
  message: string;
};

const OVERALL_REFERENCE = "__overall__";

const money = (value: number) =>
  `RM ${value.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`;

const LIAMPricingReference = () => {
  const [procedures, setProcedures] = useState<Procedure[]>([]);
  const [selection, setSelection] = useState("");
  const [facilityState, setFacilityState] = useState(OVERALL_REFERENCE);
  const [result, setResult] = useState<LIAMResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch(apiUrl("/pricing/reference/liam/procedures"), {
      signal: controller.signal,
    })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to load LIAM procedures.");
        setProcedures(Array.isArray(data.procedures) ? data.procedures : []);
      })
      .catch((fetchError) => {
        if (fetchError.name !== "AbortError") {
          setError("The LIAM procedure list could not be loaded from the API.");
        }
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selection) {
      setResult(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    const [procedureCode, careSetting] = selection.split("::");
    setLoading(true);
    setError("");
    setResult(null);

    const loadReference = async () => {
      const query = new URLSearchParams({
        procedure_code: procedureCode,
        care_setting: careSetting,
        segmentation_type:
          facilityState === OVERALL_REFERENCE ? "Overall" : "Facility State",
        segment: facilityState === OVERALL_REFERENCE ? "All" : facilityState,
      });

      try {
        const response = await fetch(
          apiUrl(`/pricing/reference/liam?${query}`),
          { signal: controller.signal },
        );
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "Unable to retrieve LIAM pricing.");
        }
        setResult(data as LIAMResult);
      } catch (fetchError) {
        if (fetchError instanceof DOMException && fetchError.name === "AbortError") {
          return;
        }
        setError(
          fetchError instanceof Error
            ? fetchError.message
            : "Cannot connect to the pricing API.",
        );
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    };

    void loadReference();
    return () => controller.abort();
  }, [facilityState, selection]);

  const selectedProcedure = procedures.find(
    (procedure) =>
      `${procedure.procedure_code}::${procedure.care_setting}` === selection,
  );
  const locationLabel =
    result?.segmentation_type === "Facility State"
      ? result.segment || facilityState
      : "Malaysia – Overall";
  const sourceHref =
    result?.source_url && result.source_pdf_page
      ? `${result.source_url}#page=${result.source_pdf_page}`
      : result?.source_url;

  return (
    <Card className="border-0 shadow-lg overflow-hidden">
      <div className="healthcare-gradient px-6 py-5 md:px-8">
        <h2 className="text-lg font-bold text-primary-foreground flex items-center gap-2">
          <BookOpen className="h-5 w-5" /> Malaysian LIAM Pricing Reference
        </h2>
        <p className="text-sm text-primary-foreground/75 mt-1">
          Published Malaysian private-healthcare percentiles; no ML model is applied.
        </p>
      </div>
      <CardContent className="p-6 md:p-8">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="liam-procedure">Procedure and care setting</Label>
            <select
              id="liam-procedure"
              required
              value={selection}
              onChange={(event) => {
                setSelection(event.target.value);
                setFacilityState(OVERALL_REFERENCE);
                setResult(null);
              }}
              className="flex h-11 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            >
              <option value="">Select a published LIAM procedure</option>
              {procedures.map((procedure) => (
                <option
                  key={`${procedure.procedure_code}-${procedure.care_setting}`}
                  value={`${procedure.procedure_code}::${procedure.care_setting}`}
                >
                  {procedure.procedure_name} — {procedure.care_setting}
                </option>
              ))}
            </select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="liam-facility-state">Facility State</Label>
            <select
              id="liam-facility-state"
              value={facilityState}
              disabled={!selectedProcedure}
              onChange={(event) => setFacilityState(event.target.value)}
              className="flex h-11 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus:border-ring focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <option value={OVERALL_REFERENCE}>Malaysia – Overall Reference</option>
              {(selectedProcedure?.facility_states || []).map((state) => (
                <option key={state.state} value={state.state}>
                  {state.state}
                  {state.data_status === "Insufficient credible data"
                    ? " — Insufficient credible data"
                    : ""}
                </option>
              ))}
            </select>
            <p className="text-xs text-muted-foreground">
              Facility State refers to the location of the private healthcare facility in LIAM&apos;s published claims data.
            </p>
          </div>

          <div className="rounded-lg border bg-muted/40 p-4 text-sm text-muted-foreground">
            LIAM is retained as a Malaysian pricing reference only. Its P25, typical, and P75 bills are not training labels for the NSS or US models.
          </div>

          {error && <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-700">{error}</div>}

          {loading && (
            <p className="text-center text-sm text-muted-foreground" role="status">
              Loading published LIAM reference...
            </p>
          )}

          {result && (
            <div className="rounded-xl bg-muted p-6 text-center">
              {result.estimate_available && result.lower_reference != null && result.typical_bill_amount != null && result.upper_reference != null ? (
                <>
                  <p className="text-xs font-semibold uppercase tracking-wider text-primary">Published LIAM Reference</p>
                  <p className="mt-1 font-semibold text-foreground">{locationLabel}</p>
                  <p className="text-sm text-muted-foreground">Published P25–P75 bill range</p>
                  <p className="mt-2 text-3xl font-bold text-primary">
                    {money(result.lower_reference)} – {money(result.upper_reference)}
                  </p>
                  <p className="mt-3 text-sm text-muted-foreground">
                    Published typical bill: {money(result.typical_bill_amount)}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    P25–P75 represents the middle 50% of published bills for the selected procedure and reference scope.
                  </p>
                  {result.number_of_discharges_band && (
                    <p className="mt-2 text-xs text-muted-foreground">
                      Reference sample: {result.number_of_discharges_band} discharges
                    </p>
                  )}
                </>
              ) : result.pricing_type === "insufficient_credible_data" ? (
                <div className="mx-auto max-w-2xl">
                  <p className="text-xs font-semibold uppercase tracking-wider text-primary">Published LIAM Reference</p>
                  <p className="mt-1 font-semibold text-foreground">{locationLabel}</p>
                  <p className="mt-4 text-lg font-bold text-amber-700">Insufficient Credible Data</p>
                  <p className="mt-2 text-sm text-muted-foreground">
                    LIAM does not provide a reliable state-specific bill range for this procedure in {locationLabel} because the available claims data were insufficient.
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Select “Malaysia – Overall Reference” to view the national published range.
                  </p>
                </div>
              ) : (
                <p className="font-semibold">Published range unavailable for this exact selection.</p>
              )}
              <p className="mt-3 text-xs text-muted-foreground">{result.message}</p>
              {sourceHref && (
                <a
                  href={sourceHref}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-3 inline-flex items-center gap-1 text-xs text-primary hover:underline"
                >
                  LIAM source PDF, page {result.source_pdf_page} <ExternalLink className="h-3 w-3" />
                </a>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default LIAMPricingReference;
