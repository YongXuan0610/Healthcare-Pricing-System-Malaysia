import { useLocation, Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Download, AlertTriangle, Building2, Building, CheckCircle2 } from 'lucide-react';
import type { HospitalPricingResult, PrivatePricingResult, PublicCharge } from '@/lib/pricingTypes';
import { formatDateDisplay } from '@/lib/utils';

type ResultSectionProps = {
  title: string;
  icon: typeof Building2;
  result?: PrivatePricingResult;
  emptyMessage: string;
};

const ResultSection = ({ title, icon: Icon, result, emptyMessage }: ResultSectionProps) => {
  return (
    <section className="rounded-xl border bg-card p-6 md:p-8 card-shadow">
      <div className="flex items-center gap-3 mb-4">
        <div className="h-10 w-10 rounded-lg bg-accent flex items-center justify-center">
          <Icon className="h-5 w-5 text-accent-foreground" />
        </div>
        <div>
          <h2 className="text-lg font-bold">{title}</h2>
          {result ? (
            <p className="text-sm text-muted-foreground">{formatDateDisplay(result.date)}</p>
          ) : (
            <p className="text-sm text-muted-foreground">No result available</p>
          )}
        </div>
      </div>

      {!result ? (
        <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground text-center">
          {emptyMessage}
        </div>
      ) : (
        <>
          <div className="text-center py-6 rounded-lg bg-muted/50 mb-6">
            <p className="text-sm text-muted-foreground mb-1">Published Reference Total</p>
            <p className="text-4xl font-extrabold text-primary">RM {result.totalCost.toLocaleString()}</p>
          </div>

          <h3 className="font-semibold mb-3">Published Price Breakdown</h3>
          <div className="space-y-3 mb-6">
            {result.breakdown.map((item, i) => (
              <div key={i} className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{item.label}</span>
                <span className="font-medium text-right">
                  {item.amount == null
                    ? 'Not included in published pricing data'
                    : <>RM {item.amount.toLocaleString()} {item.percentage ? <span className="text-xs text-muted-foreground">({item.percentage}%)</span> : null}</>}
                </span>
              </div>
            ))}
            <hr className="border-border" />
            <div className="flex items-center justify-between font-semibold">
              <span>Published Reference Total</span>
              <span>RM {result.totalCost.toLocaleString()}</span>
            </div>
          </div>

          {result.message && (
            <p className="text-xs text-muted-foreground mb-6">{result.message}</p>
          )}

          <h3 className="font-semibold mb-3">Pricing Inputs</h3>
          <div className="grid gap-2 text-sm">
            {Object.entries(result.inputs).map(([key, val]) => (
              <div key={key} className="flex justify-between">
                <span className="text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                <span className="font-medium">{val}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  );
};

const Result = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result as HospitalPricingResult | undefined;
  const historySaved = location.state?.historySaved as boolean | undefined;
  const historySaveError = location.state?.historySaveError as string | undefined;

  if (!result) {
    return (
      <div className="container py-20 text-center">
        <h2 className="text-xl font-semibold mb-4">No Pricing Result</h2>
        <p className="text-muted-foreground mb-6">Please create a pricing reference first.</p>
        <Button asChild><Link to="/predict">Go to Healthcare Cost Estimation</Link></Button>
      </div>
    );
  }

  const publicResult = result.type === 'public' ? result : undefined;
  const privateResult = result.type === 'private' ? result : undefined;

  return (
    <div className="container py-10 max-w-5xl mx-auto">
      <button onClick={() => navigate(-1)} className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-6 transition-colors">
        <ArrowLeft className="h-4 w-4" /> Back
      </button>

      <div className="grid gap-6 lg:grid-cols-2 mb-6">
        {/* Public Hospital Result */}
        <section className="rounded-xl border bg-card p-6 md:p-8 card-shadow">
          <div className="flex items-center gap-3 mb-4">
            <div className="h-10 w-10 rounded-lg bg-accent flex items-center justify-center">
              <Building2 className="h-5 w-5 text-accent-foreground" />
            </div>
            <div>
              <h2 className="text-lg font-bold">Public Hospital Cost Reference</h2>
              {publicResult ? (
                <p className="text-sm text-muted-foreground">{formatDateDisplay(publicResult.date)}</p>
              ) : (
                <p className="text-sm text-muted-foreground">No result available</p>
              )}
            </div>
          </div>

          {!publicResult ? (
            <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground text-center">
              Create a public hospital price reference to view details here.
            </div>
          ) : (
            <>
              <div className="text-center py-6 rounded-lg bg-muted/50 mb-6">
                {publicResult.estimate_available === false ? (
                  <>
                    <p className="font-semibold">Price reference unavailable for the selected service.</p>
                    <p className="text-sm text-muted-foreground mt-2">{publicResult.message}</p>
                  </>
                ) : publicResult.pricing_type === 'range' ? (
                  <>
                    <p className="text-sm text-muted-foreground mb-1">Published Price Statistics</p>
                    <p className="text-4xl font-extrabold text-primary">
                      RM {publicResult.lower_estimate.toLocaleString()} – RM {publicResult.upper_estimate.toLocaleString()}
                    </p>
                    <p className="text-sm text-muted-foreground mt-3">Typical Reference</p>
                    <p className="text-2xl font-bold">RM {publicResult.typical_estimate.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Based on {publicResult.records_used} matching published pricing records.
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-sm text-muted-foreground mb-1">Published Reference Charge</p>
                    <p className="text-4xl font-extrabold text-primary">RM {publicResult.published_cost.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground mt-2">Based on one matching published pricing record.</p>
                  </>
                )}
                {publicResult.pricing_source && (
                  <p className="text-xs text-muted-foreground mt-2 capitalize">
                    Source: {publicResult.pricing_source.replace(/_/g, ' ')}
                  </p>
                )}
                {publicResult.pricing_scope && (
                  <p className="text-xs text-muted-foreground mt-2">{publicResult.pricing_scope}</p>
                )}
              </div>

              {publicResult.estimate_available !== false && (
                <p className="text-xs text-muted-foreground mb-6">
                  This reference is based on available published Malaysian public healthcare pricing data. Actual charges may vary with treatment complexity, ward class, procedures, medication, investigations, and other applicable charges.
                </p>
              )}

              <h3 className="font-semibold mb-3">Pricing Inputs</h3>
              <div className="grid gap-2 text-sm mb-6">
                {Object.entries(publicResult.inputs).map(([key, val]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-muted-foreground capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                    <span className="font-medium">{String(val)}</span>
                  </div>
                ))}
              </div>

              {/* Matched Public Hospital Charges */}
              {publicResult.matched_public_charges && publicResult.matched_public_charges.length > 0 && (
                <>
                  <h3 className="font-semibold mb-3">Matching Published Public Charges</h3>
                  <div className="space-y-3 border-t pt-4">
                    {publicResult.matched_public_charges.map((charge: PublicCharge, idx: number) => (
                      <div key={idx} className="border rounded-lg p-3 space-y-2 text-sm">
                        <div className="flex justify-between items-start gap-2">
                          <div className="flex-1">
                            <p className="font-semibold text-foreground">{charge.service_name}</p>
                            <p className="text-xs text-muted-foreground">{charge.category}</p>
                          </div>
                          <span className="font-bold text-primary whitespace-nowrap">RM {charge.price_rm.toLocaleString()}</span>
                        </div>
                        {charge.patient_class !== 'all' && (
                          <p className="text-xs text-muted-foreground">
                            Patient Class: <span className="font-medium">{charge.patient_class}</span>
                          </p>
                        )}
                        <p className="text-xs text-muted-foreground">
                          Charge Type: <span className="font-medium">{charge.charge_type}</span>
                        </p>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
        </section>

        {/* Private Hospital Result */}
        <ResultSection
          title="Private Hospital Price Reference"
          icon={Building}
          result={privateResult}
          emptyMessage="Create a private hospital price reference to view details here."
        />
      </div>

      {historySaved && (
        <div className="rounded-lg border border-primary/20 bg-primary/5 p-4 mb-6 flex gap-3">
          <CheckCircle2 className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-semibold">Saved to prediction history</p>
            <p className="text-muted-foreground mt-1">
              This result is now available from your dashboard and Prediction History page.
            </p>
          </div>
        </div>
      )}

      {historySaveError && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 mb-6 flex gap-3">
          <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
          <div className="text-sm">
            <p className="font-semibold">Prediction history was not saved</p>
            <p className="text-muted-foreground mt-1">{historySaveError}</p>
          </div>
        </div>
      )}

      <div className="rounded-xl border bg-card p-4 md:p-5 mb-6">
        <div className="flex gap-3">
          <Button className="flex-1" asChild><Link to="/predict">New Prediction</Link></Button>
          <Button variant="outline" size="icon"><Download className="h-4 w-4" /></Button>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="rounded-lg border border-healthcare-amber/30 bg-healthcare-amber-light p-4 flex gap-3">
        <AlertTriangle className="h-5 w-5 text-healthcare-amber shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="font-semibold mb-1">Disclaimer</p>
          <p className="text-muted-foreground leading-relaxed">
            Public results are source-based charge references or statistical ranges, while private results combine only published package and ward prices. They are not guaranteed bills. Additional professional, medication, diagnostic, procedural, implant, and other charges may apply. Please confirm current pricing directly with the hospital.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Result;
