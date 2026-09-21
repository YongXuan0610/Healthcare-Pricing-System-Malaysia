import { FormEvent, useEffect, useState } from "react";
import { AlertTriangle, Database, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiUrl } from "@/lib/api";
import { supabase } from "@/lib/supabase";

export type NSSFormData = {
  age_years: string;
  number_of_hospitalisations: string;
  length_of_stay_days: string;
  gender: string;
  chronic_ailment: string;
  surgery: string;
  medicine: string;
  ailment_nature: string;
  hospitalisation_treatment_nature: string;
  medical_institution_type: string;
  ward_type: string;
  place_of_hospitalisation: string;
  pregnant: string;
  communicable_disease: string;
  other_ailment_last_15_days: string;
};

type NSSFieldKey = keyof NSSFormData;
type NSSFieldDefinition = {
  key: NSSFieldKey;
  label: string;
  type: "number" | "select";
  help?: string;
  wide?: boolean;
};

type NSSNumericLimit = { minimum: number; maximum: number; step: number };

export type NSSOptions = {
  field_order: NSSFieldKey[];
  optional_fields: NSSFieldKey[];
  numeric_limits: Partial<Record<NSSFieldKey, NSSNumericLimit>>;
  categories: Partial<Record<NSSFieldKey, string[]>>;
  model: {
    model_version: string;
    model_type: string;
    architecture: string;
    target_transform: string;
    feature_count: number;
    encoded_feature_count: number;
    r2_test: number;
    mae_test_inr: number;
    rmse_test_inr: number;
    median_absolute_error_test_inr: number;
    rmsle_test: number;
  };
  prediction_interval: { available: boolean; reason: string };
};

type NSSPrediction = {
  predicted_medical_expenditure: number;
  central_prediction?: number;
  lower_bound?: number;
  upper_bound?: number;
  interval_level?: number;
  predicted_medical_expenditure_range: { lower: number; upper: number; coverage_level: number; currency: string };
  prediction_interval_available: boolean;
  currency: string;
  limitation: string;
  range_interpretation: string;
  model_version: string;
  model_type: string;
  r2_test: number;
};

type USFormData = {
  age: string;
  sex: "female" | "male";
  bmi: string;
  children: string;
  smoker: "no" | "yes";
  region: "northeast" | "northwest" | "southeast" | "southwest";
};

type USBenchmarkPrediction = {
  predicted_charge: number;
  predicted_charge_range: {
    lower: number;
    upper: number;
    coverage_level: number;
    currency: string;
  };
  currency: string;
  limitation: string;
};

const initialNSSForm: NSSFormData = {
  age_years: "",
  number_of_hospitalisations: "",
  length_of_stay_days: "",
  gender: "",
  chronic_ailment: "",
  surgery: "",
  medicine: "",
  ailment_nature: "",
  hospitalisation_treatment_nature: "",
  medical_institution_type: "",
  ward_type: "",
  place_of_hospitalisation: "",
  pregnant: "",
  communicable_disease: "",
  other_ailment_last_15_days: "",
};

const initialUSForm: USFormData = {
  age: "35",
  sex: "male",
  bmi: "27.5",
  children: "1",
  smoker: "no",
  region: "southeast",
};

// Exported for contract tests; the runtime component remains the default export.
// eslint-disable-next-line react-refresh/only-export-components
export const NSS_FIELD_GROUPS: { title: string; fields: NSSFieldDefinition[] }[] = [
  { title: "Patient and Health Context", fields: [
      { key: "age_years", label: "Age (years)", type: "number" },
      { key: "gender", label: "Gender recorded by NSS", type: "select" },
      { key: "chronic_ailment", label: "Chronic ailment", type: "select" },
      { key: "pregnant", label: "Pregnant", type: "select" },
      { key: "communicable_disease", label: "Communicable disease", type: "select", wide: true },
      { key: "other_ailment_last_15_days", label: "Other ailment in the last 15 days", type: "select" },
    ] },
  { title: "Hospitalisation Episode", fields: [
      { key: "number_of_hospitalisations", label: "Number of hospitalisations", type: "number" },
      { key: "length_of_stay_days", label: "Length of stay (days)", type: "number" },
      { key: "ailment_nature", label: "Nature of ailment", type: "select", wide: true },
      {
        key: "hospitalisation_treatment_nature",
        label: "Nature of hospital treatment",
        type: "select",
        wide: true,
      },
      { key: "medical_institution_type", label: "Medical institution", type: "select" },
      { key: "ward_type", label: "Ward type", type: "select" },
      { key: "place_of_hospitalisation", label: "Place of hospitalisation", type: "select" },
      { key: "surgery", label: "Surgery received", type: "select" },
      { key: "medicine", label: "Medicine received", type: "select" },
    ] },
];

const numericFields = new Set<NSSFieldKey>(
  NSS_FIELD_GROUPS.flatMap((group) => group.fields)
    .filter((field) => field.type === "number")
    .map((field) => field.key),
);

// eslint-disable-next-line react-refresh/only-export-components
export const buildNSSPayload = (
  form: NSSFormData,
  optionalFields: NSSFieldKey[],
): Record<NSSFieldKey, string | number | null> => {
  const optional = new Set(optionalFields);
  return Object.fromEntries(
    NSS_FIELD_GROUPS.flatMap((group) => group.fields).map(({ key }) => {
      const value = form[key];
      if (value === "" && optional.has(key)) return [key, null];
      return [key, numericFields.has(key) ? Number(value) : value];
    }),
  ) as Record<NSSFieldKey, string | number | null>;
};

const readable = (value: string) =>
  value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

const currency = (code: string, value: number) =>
  `${code} ${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const rangeMarkerPosition = (lower: number, upper: number, central: number) => {
  if (upper <= lower) return 50;
  return Math.min(100, Math.max(0, ((central - lower) / (upper - lower)) * 100));
};

const getPredictionHeaders = async (): Promise<Record<string, string>> => {
  const {
    data: { session },
    error,
  } = await supabase.auth.getSession();

  if (error) console.error("Unable to read Supabase session:", error);

  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (session?.access_token) headers.Authorization = `Bearer ${session.access_token}`;
  return headers;
};

const AIPredictionForm = () => {
  const [nssForm, setNssForm] = useState<NSSFormData>(initialNSSForm);
  const [nssOptions, setNssOptions] = useState<NSSOptions | null>(null);
  const [nssResult, setNssResult] = useState<NSSPrediction | null>(null);
  const [nssLoading, setNssLoading] = useState(false);
  const [nssError, setNssError] = useState("");

  const [usForm, setUsForm] = useState<USFormData>(initialUSForm);
  const [usResult, setUsResult] = useState<USBenchmarkPrediction | null>(null);
  const [usLoading, setUsLoading] = useState(false);
  const [usError, setUsError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch(apiUrl("/predict/nss80/options"), { signal: controller.signal })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Unable to load NSS options.");
        setNssOptions(data as NSSOptions);
      })
      .catch((error) => {
        if (error.name !== "AbortError") {
          setNssError("The NSS practical 15 codebook options could not be loaded from the API.");
        }
      });
    return () => controller.abort();
  }, []);

  const updateNssField = (key: NSSFieldKey, value: string) => {
    setNssForm((current) => {
      const next = { ...current, [key]: value };
      if (key === "gender" || key === "age_years") next.pregnant = "";
      return next;
    });
  };

  const fieldIsApplicable = (key: NSSFieldKey) => {
    if (key === "pregnant") {
      const age = Number(nssForm.age_years);
      return nssForm.gender === "Female" && nssForm.age_years !== "" && age >= 15 && age <= 49;
    }
    return true;
  };

  const fieldIsRequired = (key: NSSFieldKey) => {
    if (!nssOptions) return true;
    if (!nssOptions.optional_fields.includes(key)) return true;
    return fieldIsApplicable(key);
  };

  const fieldOptions = (key: NSSFieldKey) => nssOptions?.categories[key] || [];

  const submitNSS = async (event: FormEvent) => {
    event.preventDefault();
    if (!nssOptions) return;
    setNssLoading(true);
    setNssError("");
    setNssResult(null);
    try {
      const headers = await getPredictionHeaders();
      const response = await fetch(apiUrl("/predict/nss80"), {
        method: "POST",
        headers,
        body: JSON.stringify(buildNSSPayload(nssForm, nssOptions.optional_fields)),
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(data.detail)
          ? data.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("; ")
          : data.detail;
        throw new Error(typeof detail === "string" ? detail : "Unable to generate NSS estimate.");
      }
      setNssResult(data as NSSPrediction);
    } catch (error) {
      setNssError(error instanceof Error ? error.message : "Cannot connect to the prediction API.");
    } finally {
      setNssLoading(false);
    }
  };

  const submitUS = async (event: FormEvent) => {
    event.preventDefault();
    setUsLoading(true);
    setUsError("");
    setUsResult(null);
    try {
      const headers = await getPredictionHeaders();
      const response = await fetch(apiUrl("/predict/benchmark/us"), {
        method: "POST",
        headers,
        body: JSON.stringify({
          ...usForm,
          age: Number(usForm.age),
          bmi: Number(usForm.bmi),
          children: Number(usForm.children),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to run the US benchmark.");
      setUsResult(data as USBenchmarkPrediction);
    } catch (error) {
      setUsError(error instanceof Error ? error.message : "Cannot connect to the prediction API.");
    } finally {
      setUsLoading(false);
    }
  };

  const renderNssField = (field: NSSFieldDefinition) => {
    const applicable = fieldIsApplicable(field.key);
    const required = fieldIsRequired(field.key);
    const commonClass = "flex h-11 w-full rounded-md border border-input bg-background px-3 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-60";
    return (
      <div key={field.key} className={`space-y-2 ${field.wide ? "sm:col-span-2" : ""}`}>
        <Label htmlFor={`nss-${field.key}`}>{field.label}</Label>
        {field.type === "number" ? (
          <Input
            id={`nss-${field.key}`}
            type="number"
            min={nssOptions?.numeric_limits[field.key]?.minimum}
            max={nssOptions?.numeric_limits[field.key]?.maximum}
            step={nssOptions?.numeric_limits[field.key]?.step || 1}
            required={required}
            disabled={!applicable}
            value={nssForm[field.key]}
            onChange={(event) => updateNssField(field.key, event.target.value)}
          />
        ) : (
          <select
            id={`nss-${field.key}`}
            required={required}
            disabled={!applicable}
            value={nssForm[field.key]}
            onChange={(event) => updateNssField(field.key, event.target.value)}
            className={commonClass}
          >
            <option value="" disabled={required}>
              {applicable && required ? `Select ${field.label.toLowerCase()}` : "Not applicable / not recorded"}
            </option>
            {fieldOptions(field.key).map((option) => (
              <option key={option} value={option}>{readable(option)}</option>
            ))}
          </select>
        )}
        {field.help && <p className="text-xs text-muted-foreground">{field.help}</p>}
      </div>
    );
  };

  return (
    <Card className="border-0 shadow-lg overflow-hidden">
      <div className="healthcare-gradient px-6 py-5 md:px-8">
        <h2 className="text-lg font-bold text-primary-foreground flex items-center gap-2">
          <FlaskConical className="h-5 w-5" /> Prediction Cost / Individual Cost
        </h2>
        <p className="mt-1 max-w-4xl text-sm leading-relaxed text-primary-foreground/75">
          Explore research estimates while every model retains its original dataset, country, and currency.
        </p>
      </div>

      <CardContent className="p-6 md:p-8">
        <Tabs defaultValue="nss80">
          <TabsList className="grid w-full grid-cols-2 mb-8">
            <TabsTrigger value="nss80">NSS 80 Primary</TabsTrigger>
            <TabsTrigger value="us">US Benchmark</TabsTrigger>
          </TabsList>

          <TabsContent value="nss80">
            <form onSubmit={submitNSS} className="space-y-7">
              <div className="rounded-lg border bg-muted/40 p-4 flex gap-3">
                <Database className="h-5 w-5 text-primary shrink-0 mt-0.5" />
                <div className="text-sm text-muted-foreground">
                  <h3 className="font-semibold text-foreground">NSS 80 Healthcare Expenditure Model</h3>
                  <p className="mt-1">
                    Uses 15 leakage-safe patient and hospitalisation inputs to estimate medical expenditure for an inpatient case in India. Results are shown in INR.
                  </p>
                </div>
              </div>

              {NSS_FIELD_GROUPS.map((group) => (
                <section key={group.title} className="space-y-4 rounded-xl border p-4 md:p-5">
                  <h3 className="font-semibold text-foreground">{group.title}</h3>
                  <div className="grid gap-4 sm:grid-cols-2">
                    {group.fields.map(renderNssField)}
                  </div>
                </section>
              ))}

              {nssError && (
                <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-700">
                  {nssError}
                </div>
              )}

              {nssResult && (
                <div className="rounded-xl bg-muted p-6 text-center">
                  <h3 className="text-sm font-medium text-muted-foreground">Estimated Medical Expenditure Range</h3>
                  <p className="mt-2 text-3xl font-bold text-primary">
                    {currency(nssResult.currency, nssResult.predicted_medical_expenditure_range.lower)}
                    {" – "}
                    {currency(nssResult.currency, nssResult.predicted_medical_expenditure_range.upper)}
                  </p>
                  <p className="mt-3 text-sm text-muted-foreground">
                    Estimated {Math.round(nssResult.predicted_medical_expenditure_range.coverage_level * 100)}% prediction range.
                  </p>
                  <div
                    aria-label="Prediction range visualization"
                    className="relative mx-auto mt-5 h-2 max-w-xl rounded-full bg-primary/20"
                  >
                    <span className="absolute inset-y-0 left-0 right-0 rounded-full bg-primary/45" />
                    <span
                      aria-label="Central model estimate marker"
                      className="absolute top-1/2 h-4 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary"
                      style={{ left: `${rangeMarkerPosition(
                        nssResult.predicted_medical_expenditure_range.lower,
                        nssResult.predicted_medical_expenditure_range.upper,
                        nssResult.predicted_medical_expenditure,
                      )}%` }}
                    />
                  </div>
                  <div className="mx-auto mt-2 flex max-w-xl justify-between text-xs text-muted-foreground">
                    <span>Lower estimate</span><span>Upper estimate</span>
                  </div>
                  <p className="mt-4 text-sm text-muted-foreground">
                    Central model estimate: {currency(nssResult.currency, nssResult.predicted_medical_expenditure)}
                  </p>
                  <p className="mt-3 text-sm text-muted-foreground">
                    Actual expenditure may vary. Based on India’s NSS 80 dataset; not a Malaysian hospital quotation.
                  </p>
                </div>
              )}

              <Button type="submit" disabled={nssLoading || !nssOptions} className="w-full h-12">
                {nssLoading ? "Generating NSS Estimate..." : "Generate NSS Primary Estimate"}
              </Button>
            </form>
          </TabsContent>

          <TabsContent value="us">
            <form onSubmit={submitUS} className="space-y-7">
              <div className="rounded-lg border border-healthcare-amber/30 bg-healthcare-amber-light p-4 flex gap-3">
                <AlertTriangle className="h-5 w-5 text-healthcare-amber shrink-0 mt-0.5" />
                <p className="text-sm text-muted-foreground">
                  Secondary benchmark only. It uses 1,338 US Kaggle records, stays in USD, and is not combined with NSS or Malaysian pricing.
                </p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                {[
                  { key: "age", label: "Age", min: 18, max: 64, step: 1 },
                  { key: "bmi", label: "BMI", min: 15.96, max: 53.13, step: 0.01 },
                  { key: "children", label: "Number of Children", min: 0, max: 5, step: 1 },
                ].map((field) => (
                  <div key={field.key} className="space-y-2">
                    <Label htmlFor={`us-${field.key}`}>{field.label}</Label>
                    <Input id={`us-${field.key}`} type="number" min={field.min} max={field.max} step={field.step} required
                      value={usForm[field.key as keyof USFormData]}
                      onChange={(event) => setUsForm((current) => ({ ...current, [field.key]: event.target.value }))} />
                  </div>
                ))}

                {[
                  { key: "sex", label: "Sex", options: ["female", "male"] },
                  { key: "smoker", label: "Smoking Status", options: ["no", "yes"] },
                  { key: "region", label: "US Dataset Region", options: ["northeast", "northwest", "southeast", "southwest"] },
                ].map((field) => (
                  <div key={field.key} className="space-y-2">
                    <Label htmlFor={`us-${field.key}`}>{field.label}</Label>
                    <select id={`us-${field.key}`} value={usForm[field.key as keyof USFormData]}
                      onChange={(event) => setUsForm((current) => ({ ...current, [field.key]: event.target.value }))}
                      className="flex h-11 w-full rounded-md border border-input bg-background px-3 py-2 text-sm">
                      {field.options.map((option) => <option key={option} value={option}>{readable(option)}</option>)}
                    </select>
                  </div>
                ))}
              </div>

              {usError && <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-700">{usError}</div>}

              {usResult && (
                <div className="rounded-xl bg-muted p-6 text-center">
                  <p className="text-sm text-muted-foreground">Secondary benchmark result</p>
                  <p className="mt-2 text-3xl font-bold text-primary">
                    {currency(usResult.currency, usResult.predicted_charge_range.lower)} – {currency(usResult.currency, usResult.predicted_charge_range.upper)}
                  </p>
                  <p className="mt-3 text-sm text-muted-foreground">Point estimate: {currency(usResult.currency, usResult.predicted_charge)}</p>
                  <p className="mt-3 text-xs text-muted-foreground">{usResult.limitation}</p>
                </div>
              )}

              <Button type="submit" disabled={usLoading} className="w-full h-12" variant="outline">
                {usLoading ? "Running Benchmark..." : "Run US Secondary Benchmark"}
              </Button>
            </form>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default AIPredictionForm;
