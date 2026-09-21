import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Building2, Building, FlaskConical, Trash2 } from 'lucide-react';
import PageHeader from '@/components/PageHeader';
import { formatDateDisplay } from '@/lib/utils';
import { supabase } from '@/lib/supabase';

type PredictionInput = Record<string, unknown>;

type PredictionBreakdownItem = {
  label: string;
  amount: number;
};

type PredictionHistoryRow = {
  id: string;
  user_id: string;
  hospital_type: 'public' | 'private' | 'nss80';
  total_cost: number | string;
  currency: string;
  inputs: PredictionInput | null;
  breakdown: unknown;
  created_at: string;
};

const PredictionHistory = () => {
  const navigate = useNavigate();

  const [predictions, setPredictions] = useState<PredictionHistoryRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const loadPredictions = async () => {
      let isRedirecting = false;
      setIsLoading(true);
      setErrorMessage('');

      try {
        const {
          data: { user },
          error: userError,
        } = await supabase.auth.getUser();

        if (!user) {
          isRedirecting = true;
          navigate('/login', { replace: true });
          return;
        }

        if (userError) {
          throw userError;
        }

        const { data, error } = await supabase
          .from('prediction_history')
          .select(
            'id, user_id, hospital_type, total_cost, currency, inputs, breakdown, created_at',
          )
          .eq('user_id', user.id)
          .order('created_at', { ascending: false });

        if (error) {
          throw error;
        }

        if (!isMounted) return;

        setPredictions((data ?? []) as PredictionHistoryRow[]);
      } catch (error) {
        console.error('Prediction history loading error:', error);

        if (isMounted) {
          setErrorMessage(
            'Unable to load your prediction history. Please try again.',
          );
        }
      } finally {
        if (isMounted && !isRedirecting) {
          setIsLoading(false);
        }
      }
    };

    void loadPredictions();

    return () => {
      isMounted = false;
    };
  }, [navigate]);

  const formatInputLabel = (key: string) =>
    key
      .replace(/_/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/\b\w/g, (character) => character.toUpperCase());

  const formatInputValue = (value: unknown) => {
    if (value === null || value === undefined || value === '') {
      return '—';
    }

    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }

    if (Array.isArray(value)) {
      return value.map((item) => String(item)).join(', ');
    }

    if (typeof value === 'object') {
      return JSON.stringify(value);
    }

    return String(value);
  };

  const formatMoney = (amount: number | string, currency = 'MYR') => {
    const numericAmount = Number(amount);

    if (!Number.isFinite(numericAmount)) {
      return `${currency} ${amount}`;
    }

    if (currency === 'MYR') {
      return `RM ${numericAmount.toLocaleString('en-MY', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`;
    }

    return `${currency} ${numericAmount.toLocaleString('en-MY', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  const getBreakdownItems = (
    breakdown: unknown,
  ): PredictionBreakdownItem[] => {
    if (!Array.isArray(breakdown)) {
      return [];
    }

    return breakdown.flatMap((item) => {
      if (
        typeof item === 'object' &&
        item !== null &&
        'label' in item &&
        'amount' in item
      ) {
        const label = String(item.label);
        const amount = Number(item.amount);

        if (Number.isFinite(amount)) {
          return [{ label, amount }];
        }
      }

      return [];
    });
  };

  const getNssRange = (breakdown: unknown) => {
    if (typeof breakdown !== 'object' || breakdown === null || Array.isArray(breakdown)) {
      return null;
    }
    const record = breakdown as Record<string, unknown>;
    const lower = Number(record.lower_bound);
    const upper = Number(record.upper_bound);
    const level = Number(record.interval_level);
    if (![lower, upper, level].every(Number.isFinite) || lower < 0 || upper < lower) {
      return null;
    }
    return { lower, upper, level };
  };

  const handleDelete = async (predictionId: string) => {
    const confirmed = window.confirm(
      'Are you sure you want to delete this prediction from your history?',
    );

    if (!confirmed) {
      return;
    }

    setDeletingId(predictionId);
    setErrorMessage('');

    try {
      const { error } = await supabase
        .from('prediction_history')
        .delete()
        .eq('id', predictionId);

      if (error) {
        throw error;
      }

      setPredictions((current) =>
        current.filter((prediction) => prediction.id !== predictionId),
      );
    } catch (error) {
      console.error('Prediction deletion error:', error);
      setErrorMessage('Unable to delete this prediction. Please try again.');
    } finally {
      setDeletingId(null);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-muted/30 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">
          Loading prediction history...
        </p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-muted/30">
      <header className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur-md">
        <div className="container flex h-16 items-center">
          <Link
            to="/dashboard"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
        </div>
      </header>

      <PageHeader
        title="Prediction History"
        description="View and manage all your past predictions."
      />

      <div className="container py-8">
        {errorMessage && (
          <div
            className="mb-6 max-w-3xl rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
            role="alert"
          >
            {errorMessage}
          </div>
        )}

        {predictions.length === 0 ? (
          <div className="text-center py-16">
            <p className="font-medium mb-1">No predictions yet.</p>
            <p className="text-sm text-muted-foreground mb-4">
              Your saved healthcare cost estimations will appear here.
            </p>
            <Button asChild>
              <Link to="/predict">Make Your First Prediction</Link>
            </Button>
          </div>
        ) : (
          <div className="space-y-4 max-w-3xl">
            {predictions.map((prediction) => {
              const Icon = prediction.hospital_type === 'public'
                ? Building2
                : prediction.hospital_type === 'nss80'
                  ? FlaskConical
                  : Building;

              const inputEntries = Object.entries(prediction.inputs ?? {});
              const breakdownItems = getBreakdownItems(prediction.breakdown);
              const nssRange = getNssRange(prediction.breakdown);

              return (
                <div
                  key={prediction.id}
                  className="rounded-xl border bg-card p-5 card-shadow"
                >
                  <div className="flex items-start justify-between gap-4 mb-3">
                    <div className="flex items-center gap-3">
                      <div className="h-9 w-9 rounded-lg bg-accent flex items-center justify-center">
                        <Icon className="h-4 w-4 text-accent-foreground" />
                      </div>

                      <div>
                        <p className="font-semibold text-sm">
                          {prediction.hospital_type === 'nss80'
                            ? 'NSS 80 Research Estimate'
                            : `${prediction.hospital_type} Hospital`}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {formatDateDisplay(prediction.created_at)}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <p className="text-lg font-bold text-primary whitespace-nowrap">
                        {nssRange
                          ? `${formatMoney(nssRange.lower, prediction.currency)} – ${formatMoney(nssRange.upper, prediction.currency)}`
                          : formatMoney(prediction.total_cost, prediction.currency)}
                      </p>

                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDelete(prediction.id)}
                        disabled={deletingId === prediction.id}
                        aria-label="Delete prediction"
                        title="Delete prediction"
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </div>

                  {inputEntries.length > 0 && (
                    <div className="grid gap-1 text-sm">
                      {inputEntries.map(([key, value]) => (
                        <div
                          key={key}
                          className="flex justify-between gap-4"
                        >
                          <span className="text-muted-foreground">
                            {formatInputLabel(key)}
                          </span>
                          <span className="font-medium text-right">
                            {formatInputValue(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="mt-3 pt-3 border-t">
                    {nssRange ? (
                      <p className="text-xs text-muted-foreground">
                        Estimated {Math.round(nssRange.level * 100)}% prediction range · Central model estimate: {formatMoney(prediction.total_cost, prediction.currency)}
                      </p>
                    ) : breakdownItems.length > 0 ? (
                      <p className="text-xs text-muted-foreground">
                        {breakdownItems
                          .map(
                            (item) =>
                              `${item.label}: ${formatMoney(
                                item.amount,
                                prediction.currency,
                              )}`,
                          )
                          .join(' • ')}
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        No detailed cost breakdown was saved for this
                        prediction.
                      </p>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionHistory;
