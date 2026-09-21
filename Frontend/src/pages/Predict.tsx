import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import PageHeader from '@/components/PageHeader';
import AIPredictionForm from '@/components/AIPredictionForm';
import HealthcareServiceAssistant from '@/components/HealthcareServiceAssistant';
import LIAMPricingReference from '@/components/LIAMPricingReference';
import { apiUrl, type HealthcareServiceSuggestion } from '@/lib/api';
import {
  hospitalPageSizeForViewport,
  paginateHospitals,
} from '@/lib/hospitalPagination';
import { supabase } from '@/lib/supabase';
import { useIsMobile } from '@/hooks/use-mobile';
import type {
  PublicHospitalOption,
  PublicHospitalPricing,
  PublicNationalReference,
  PublicPricingApiResponse,
} from '@/lib/pricingTypes';
import { cn, formatDateDisplay } from '@/lib/utils';

import {
  Building2,
  Building,
  Stethoscope,
  Users,
  Package,
  BedDouble,
  CalendarDays,
  Calculator,
  Info,
  ChevronLeft,
  ChevronRight,
  FlaskConical,
  BookOpen,
  ExternalLink,
} from "lucide-react";

type PrivatePackageOption = {
  id: string;
  hospital: string;
  name: string;
  price: number;
  description: string;
  gender_target?: string;
  age_target?: string;
};

type PrivateWardRateOption = {
  id: string;
  name: string;
  dailyRate: number;
  description: string;
  rateBasis?: string;
  notes?: string;
};

type PredictionHistoryBreakdownItem = {
  label: string;
  amount: number;
};

type PredictionHistoryPayload = {
  type: 'public' | 'private';
  totalCost: number;
  currency?: string;
  inputs: Record<string, unknown>;
  breakdown: PredictionHistoryBreakdownItem[];
};

type PublicSystemSettings = {
  system_name: string;
  contact_email: string;
  allow_guest_predictions: boolean;
  maintenance_mode: boolean;
};

type PricingAccessStatus =
  | 'checking'
  | 'allowed'
  | 'maintenance'
  | 'redirecting'
  | 'error';

const formatPublishedPrice = (price: number): string =>
  price === 0 ? 'Free' : `RM ${price.toLocaleString()}`;

const getPredictionRequestHeaders = async (): Promise<Record<string, string>> => {
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (session?.access_token) {
    headers.Authorization = `Bearer ${session.access_token}`;
  }

  return headers;
};

const savePredictionHistory = async (
  prediction: PredictionHistoryPayload,
): Promise<{ saved: boolean; error?: string }> => {
  const {
    data: { user },
    error: userError,
  } = await supabase.auth.getUser();

  // Guest users can still generate a result; their history is simply not saved.
  if (!user) {
    if (userError) {
      console.error('Unable to read Supabase user:', userError);
    }
    return { saved: false };
  }

  const totalCost = Number(prediction.totalCost);

  if (!Number.isFinite(totalCost) || totalCost < 0) {
    return {
      saved: false,
      error: 'The result was generated, but it did not contain a valid cost to save.',
    };
  }

  const { error } = await supabase.from('prediction_history').insert({
    user_id: user.id,
    hospital_type: prediction.type,
    total_cost: totalCost,
    currency: prediction.currency || 'MYR',
    inputs: prediction.inputs,
    breakdown: prediction.breakdown,
  });

  if (error) {
    console.error('Prediction history save error:', error);
    return {
      saved: false,
      error: 'The result was generated, but it could not be saved to your history.',
    };
  }

  return { saved: true };
};

const Predict = () => {
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  const [tab, setTab] = useState('public');
  const [pricingAccessStatus, setPricingAccessStatus] =
    useState<PricingAccessStatus>('checking');
  const [systemSettings, setSystemSettings] =
    useState<PublicSystemSettings | null>(null);
  const [pricingAccessError, setPricingAccessError] = useState('');

  // Public form
  const [pubCategory, setPubCategory] = useState('');
  const [pubCitizen, setPubCitizen] = useState('');
  const [pubHospital, setPubHospital] = useState('all');
  const [pubHospitals, setPubHospitals] = useState<PublicHospitalPricing[]>([]);
  const [pubNationalReferences, setPubNationalReferences] = useState<PublicNationalReference[]>([]);
  const [publicHospitals, setPublicHospitals] = useState<PublicHospitalOption[]>([]);
  const [publicHospitalsLoading, setPublicHospitalsLoading] = useState(false);
  const [publicHospitalSourcesChecked, setPublicHospitalSourcesChecked] = useState(0);
  const [currentHospitalPage, setCurrentHospitalPage] = useState(1);
  const [pubEstimateLoading, setPubEstimateLoading] = useState(false);
  const [publicCategories, setPublicCategories] = useState<string[]>([]);
  const [publicCategoriesLoading, setPublicCategoriesLoading] = useState(false);
  const [publicCategoriesError, setPublicCategoriesError] = useState('');
  const [publicSubmissionError, setPublicSubmissionError] = useState('');
  const [pubServiceFilter, setPubServiceFilter] =
    useState<HealthcareServiceSuggestion | null>(null);

  // Private form
  const [privHospital, setPrivHospital] = useState('');
  const [privPackage, setPrivPackage] = useState('');
  const [privWard, setPrivWard] = useState('');
  const [privNights, setPrivNights] = useState('1');
  const [privateHospitals, setPrivateHospitals] = useState<string[]>([]);
  const [hospitalPackages, setHospitalPackages] = useState<PrivatePackageOption[]>([]);
  const [hospitalWardRates, setHospitalWardRates] = useState<PrivateWardRateOption[]>([]);
  const [privateOptionsLoading, setPrivateOptionsLoading] = useState(false);
  const [privateOptionsError, setPrivateOptionsError] = useState('');
  const [wardRatesLoading, setWardRatesLoading] = useState(false);
  const [wardRatesError, setWardRatesError] = useState('');
  const [privateSubmissionError, setPrivateSubmissionError] = useState('');
  const hasNoStay = privWard === 'No Stay';

  const selectedPkg = hospitalPackages.find((p) => p.id === privPackage);
  const hospitalPageSize = hospitalPageSizeForViewport(isMobile);
  const hospitalPagination = useMemo(
    () => paginateHospitals(
      pubHospitals,
      currentHospitalPage,
      hospitalPageSize,
    ),
    [currentHospitalPage, hospitalPageSize, pubHospitals],
  );

  useEffect(() => {
    if (currentHospitalPage !== hospitalPagination.currentPage) {
      setCurrentHospitalPage(hospitalPagination.currentPage);
    }
  }, [currentHospitalPage, hospitalPagination.currentPage]);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    const verifyPricingAccess = async () => {
      setPricingAccessStatus('checking');
      setPricingAccessError('');

      try {
        const [settingsResponse, sessionResult] = await Promise.all([
          fetch(apiUrl('/system/settings'), {
            signal: controller.signal,
          }),
          supabase.auth.getSession(),
        ]);

        const settingsData = await settingsResponse.json();

        if (!settingsResponse.ok) {
          throw new Error(
            settingsData.detail || 'Unable to load system settings.',
          );
        }

        const settings: PublicSystemSettings = {
          system_name:
            typeof settingsData.system_name === 'string'
              ? settingsData.system_name
              : 'MyCareCost',
          contact_email:
            typeof settingsData.contact_email === 'string'
              ? settingsData.contact_email
              : '',
          allow_guest_predictions:
            settingsData.allow_guest_predictions !== false,
          maintenance_mode: settingsData.maintenance_mode === true,
        };

        if (!isMounted) {
          return;
        }

        setSystemSettings(settings);

        if (sessionResult.error) {
          console.error(
            'Unable to read Supabase session:',
            sessionResult.error,
          );
        }

        const session = sessionResult.data.session;

        if (!settings.allow_guest_predictions && !session) {
          setPricingAccessStatus('redirecting');
          navigate('/login', { replace: true });
          return;
        }

        let role = 'guest';

        if (session?.user) {
          const { data: profile, error: profileError } = await supabase
            .from('profiles')
            .select('role')
            .eq('id', session.user.id)
            .maybeSingle();

          if (profileError) {
            console.error('Unable to read profile role:', profileError);
          } else if (profile?.role) {
            role = profile.role;
          }
        }

        if (!isMounted) {
          return;
        }

        if (settings.maintenance_mode && role !== 'admin') {
          setPricingAccessStatus('maintenance');
          return;
        }

        setPricingAccessStatus('allowed');
      } catch (error) {
        if (
          error instanceof DOMException &&
          error.name === 'AbortError'
        ) {
          return;
        }

        console.error('Pricing access verification error:', error);

        if (isMounted) {
          setPricingAccessError(
            'Unable to verify whether the pricing tools are currently available.',
          );
          setPricingAccessStatus('error');
        }
      }
    };

    void verifyPricingAccess();

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [navigate]);

  const handlePredictionAccessResponse = (
    response: Response,
    detail?: unknown,
  ): boolean => {
    const message =
      typeof detail === 'string' ? detail : '';

    if (response.status === 401) {
      setPricingAccessStatus('redirecting');
      navigate('/login', { replace: true });
      return true;
    }

    if (
      response.status === 503 &&
      message.toLowerCase().includes('maintenance')
    ) {
      setPricingAccessStatus('maintenance');
      return true;
    }

    return false;
  };

  useEffect(() => {
    if (pricingAccessStatus !== 'allowed') {
      return;
    }

    const controller = new AbortController();

    setPublicCategoriesLoading(true);
    setPublicCategoriesError('');

    fetch(apiUrl('/public/categories'), {
      signal: controller.signal,
    })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Unable to load public pricing categories.');
        }

        const categories = Array.isArray(data.categories)
          ? data.categories.filter(
              (category: unknown): category is string =>
                typeof category === 'string' && category.trim().length > 0,
            )
          : [];

        setPublicCategories(categories);

        if (categories.length === 0) {
          setPublicCategoriesError(
            'No public pricing categories are currently available.',
          );
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Error fetching public pricing categories:', error);
          setPublicCategoriesError(
            'Public pricing categories are currently unavailable.',
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPublicCategoriesLoading(false);
        }
      });

    return () => controller.abort();
  }, [pricingAccessStatus]);

  useEffect(() => {
    if (pricingAccessStatus !== 'allowed') return;
    const controller = new AbortController();
    setPublicHospitalsLoading(true);

    fetch(apiUrl('/public/hospitals'), { signal: controller.signal })
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Unable to load public hospitals.');
        }
        setPublicHospitals(Array.isArray(data.hospitals) ? data.hospitals : []);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Error fetching public hospitals:', error);
          setPublicSubmissionError('Public hospital options are currently unavailable.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setPublicHospitalsLoading(false);
      });

    return () => controller.abort();
  }, [pricingAccessStatus]);

  useEffect(() => {
    if (pricingAccessStatus !== 'allowed') {
      return;
    }

    const controller = new AbortController();

    setPrivateOptionsLoading(true);
    setPrivateOptionsError('');

    fetch(apiUrl('/private/hospitals'), {
      signal: controller.signal,
    })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Unable to load private hospitals.');
        }

        setPrivateHospitals(data.hospitals || []);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Error fetching private hospitals:', error);
          setPrivateOptionsError(
            'Private hospital pricing options are currently unavailable.',
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPrivateOptionsLoading(false);
        }
      });

    return () => controller.abort();
  }, [pricingAccessStatus]);

  useEffect(() => {
    if (pricingAccessStatus !== 'allowed') {
      return;
    }

    setPrivPackage('');
    setHospitalPackages([]);
    setPrivateOptionsError('');

    if (!privHospital) {
      return;
    }

    const controller = new AbortController();

    setPrivateOptionsLoading(true);

    fetch(apiUrl(`/private/packages/${encodeURIComponent(privHospital)}`), {
      signal: controller.signal,
    })
      .then(async (response) => {
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Unable to load hospital packages.');
        }

        setHospitalPackages(data.packages || []);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          console.error('Error fetching hospital packages:', error);
          setPrivateOptionsError(
            'Hospital packages are unavailable for this hospital.',
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setPrivateOptionsLoading(false);
        }
      });

    return () => controller.abort();
  }, [privHospital, pricingAccessStatus]);

  useEffect(() => {
    if (pricingAccessStatus !== 'allowed') {
      return;
    }

    setPrivWard('');
    setPrivNights('1');
    setHospitalWardRates([]);
    setWardRatesError('');

    if (!privHospital) return;

    const controller = new AbortController();
    setWardRatesLoading(true);

    fetch(apiUrl(`/private/wards/${encodeURIComponent(privHospital)}`), {
      signal: controller.signal,
    })
      .then(async response => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || 'Unable to load ward rates.');
        setHospitalWardRates(data.ward_rates || []);
      })
      .catch(error => {
        if (error.name !== 'AbortError') {
          console.error('Error fetching private hospital ward rates:', error);
          setWardRatesError('Ward rates are unavailable for this hospital.');
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setWardRatesLoading(false);
      });

    return () => controller.abort();
  }, [privHospital, pricingAccessStatus]);

  // Live cost preview for private
  const liveCost = useMemo(() => {
    if (!selectedPkg || !privWard) return null;

    const ward = hospitalWardRates.find((w) => w.name === privWard);
    const parsedNights = parseInt(privNights);
    const nights = Number.isNaN(parsedNights) ? 1 : parsedNights;

    if (!ward) return null;

    const packageCost = selectedPkg.price;
    const wardCost = ward.dailyRate * nights;

    return {
      packageCost,
      wardCost,
      total: packageCost + wardCost,
    };
  }, [selectedPkg, privWard, privNights, hospitalWardRates]);

  // Fetch matched charges as user selects options
  const fetchPublicMatches = async (
    category: string,
    citizenship: string,
    serviceName?: string | null,
    hospitalCode: string = pubHospital,
  ) => {
    setCurrentHospitalPage(1);

    if (!category || !citizenship) {
      setPubHospitals([]);
      setPubNationalReferences([]);
      return;
    }

    setPubEstimateLoading(true);
    try {
      const response = await fetch(apiUrl("/predict/public"), {
        method: "POST",
        headers: await getPredictionRequestHeaders(),
        body: JSON.stringify({
          category,
          citizenship,
          hospital: hospitalCode,
          service_name: serviceName || undefined,
        }),
      });

      const data = (await response.json()) as PublicPricingApiResponse & {
        detail?: string;
      };

      if (!response.ok) {
        if (handlePredictionAccessResponse(response, data.detail)) {
          setPubHospitals([]);
          setPubNationalReferences([]);
          return;
        }

        throw new Error(data.detail || 'Unable to load public pricing matches.');
      }

      setPubHospitals(Array.isArray(data.hospitals) ? data.hospitals : []);
      setPubNationalReferences(
        Array.isArray(data.national_references) ? data.national_references : [],
      );
      setPublicHospitalSourcesChecked(Number(data.hospital_sources_checked) || 0);
    } catch (error) {
      console.error("Error fetching public matches:", error);
      setPubHospitals([]);
      setPubNationalReferences([]);
    } finally {
      setPubEstimateLoading(false);
    }
  };

  const handleUseAssistantService = (
    suggestion: HealthcareServiceSuggestion,
  ) => {
    const availableCategory = publicCategories.find(
      (category) => category.toLowerCase() === suggestion.category.toLowerCase(),
    );

    if (!availableCategory) {
      setPublicSubmissionError(
        'This suggested service is not currently available in the public pricing categories. Please select a category manually.',
      );
      return;
    }

    setPublicSubmissionError('');
    setTab('public');
    setPubCategory(availableCategory);
    setPubServiceFilter(suggestion);
    void fetchPublicMatches(
      availableCategory,
      pubCitizen,
      suggestion.service_name,
    );
  };

  const clearAssistantServiceFilter = () => {
    setPubServiceFilter(null);
    void fetchPublicMatches(pubCategory, pubCitizen);
  };

  const handlePublicSubmit = async () => {
    setPublicSubmissionError('');
    if (!pubCategory || !pubCitizen) return;

    try {
      const response = await fetch(apiUrl("/predict/public"), {
        method: "POST",
        headers: await getPredictionRequestHeaders(),
        body: JSON.stringify({
          category: pubCategory,
          citizenship: pubCitizen,
          hospital: pubHospital,
          service_name: pubServiceFilter?.service_name || undefined,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        if (handlePredictionAccessResponse(response, data.detail)) {
          return;
        }

        throw new Error(
          data.detail || 'Unable to generate the public price reference.',
        );
      }

      const result = {
        type: "public",
        date: formatDateDisplay(new Date()),
        totalCost: data.predicted_cost,
        estimate_available: data.estimate_available,
        pricing_type: data.pricing_type,
        published_cost: data.published_cost,
        lower_estimate: data.lower_estimate,
        typical_estimate: data.typical_estimate,
        upper_estimate: data.upper_estimate,
        records_used: data.records_used,
        range_method: data.range_method,
        currency: data.currency,
        message: data.message,
        pricing_scope: data.pricing_scope,
        methodology: data.methodology,
        inputs: {
          category: pubCategory,
          ...(pubServiceFilter
            ? { service: pubServiceFilter.mapped_service }
            : {}),
          citizenship: pubCitizen,
          hospital:
            pubHospital === 'all'
              ? 'All Public Hospitals'
              : publicHospitals.find((item) => item.hospital_code === pubHospital)
                  ?.hospital_name || pubHospital,
        },
        breakdown: [],
        pricing_source: data.pricing_source,
        matched_public_charges: data.matched_public_charges || [],
        hospitals: data.hospitals || [],
        national_references: data.national_references || [],
        lookup_summary: data.lookup_summary,
      };

      let historyStatus: { saved: boolean; error?: string } = { saved: false };

      if (
        data.estimate_available !== false &&
        Number.isFinite(Number(data.predicted_cost))
      ) {
        const publicBreakdown: PredictionHistoryBreakdownItem[] = [];

        if (data.pricing_type === 'range') {
          if (Number.isFinite(Number(data.lower_estimate))) {
            publicBreakdown.push({
              label: 'Lower Published Reference',
              amount: Number(data.lower_estimate),
            });
          }
          if (Number.isFinite(Number(data.typical_estimate))) {
            publicBreakdown.push({
              label: 'Typical Reference',
              amount: Number(data.typical_estimate),
            });
          }
          if (Number.isFinite(Number(data.upper_estimate))) {
            publicBreakdown.push({
              label: 'Upper Published Reference',
              amount: Number(data.upper_estimate),
            });
          }
        } else if (Number.isFinite(Number(data.published_cost))) {
          publicBreakdown.push({
            label: 'Published Reference Charge',
            amount: Number(data.published_cost),
          });
        }

        historyStatus = await savePredictionHistory({
          type: 'public',
          totalCost: Number(data.predicted_cost),
          currency: data.currency || 'MYR',
          inputs: result.inputs,
          breakdown: publicBreakdown,
        });
      }

      navigate("/result", {
        state: {
          result,
          historySaved: historyStatus.saved,
          historySaveError: historyStatus.error,
        },
      });
    } catch (error) {
      console.error("Public prediction error:", error);
      const message = error instanceof Error ? error.message.trim() : '';
      setPublicSubmissionError(
        message && !/failed to fetch|networkerror|load failed/i.test(message)
          ? message
          : 'Unable to generate the public price reference. Please try again.',
      );
    }
  };

  const handlePrivateSubmit = async () => {
    setPrivateSubmissionError('');
    if (!privHospital || !privPackage || !privWard || !privNights) return;

    try {
      const selectedPackageName = selectedPkg?.name ?? privPackage;
      const response = await fetch(apiUrl("/predict/private"), {
        method: "POST",
        headers: await getPredictionRequestHeaders(),
        body: JSON.stringify({
          hospital: privHospital,
          package_name: selectedPackageName,
          ward_type: privWard,
          nights: Number(privNights),
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        if (handlePredictionAccessResponse(response, data.detail)) {
          return;
        }

        throw new Error(
          data.detail || data.error || 'Unable to generate private prediction.',
        );
      }

      if (data.error) {
        throw new Error(data.error || 'Unable to generate private prediction.');
      }

      const result = {
        type: "private",
        date: formatDateDisplay(new Date()),
        totalCost: data.predicted_cost,
        inputs: {
          hospital: privHospital,
          package: selectedPackageName,
          ward: privWard,
          nights: privNights,
        },
        breakdown: [
          { label: "Published Package Price", amount: data.breakdown?.package_price ?? selectedPkg?.price ?? 0, percentage: 0 },
          { label: "Published Ward Cost", amount: data.breakdown?.ward_cost ?? 0, percentage: 0 },
          { label: "Surgeon Fee", amount: data.breakdown?.surgeon_fee ?? null, percentage: null },
          { label: "Miscellaneous Fee", amount: data.breakdown?.misc_fee ?? null, percentage: null },
        ],
        matching_private_packages: data.matching_private_packages,
        message: data.message,
      };

      const privateBreakdown: PredictionHistoryBreakdownItem[] = result.breakdown
        .filter(
          (item) =>
            item.amount !== null &&
            Number.isFinite(Number(item.amount)),
        )
        .map((item) => ({
          label: item.label,
          amount: Number(item.amount),
        }));

      const historyStatus = await savePredictionHistory({
        type: 'private',
        totalCost: Number(data.predicted_cost),
        currency: data.currency || 'MYR',
        inputs: result.inputs,
        breakdown: privateBreakdown,
      });

      navigate("/result", {
        state: {
          result,
          historySaved: historyStatus.saved,
          historySaveError: historyStatus.error,
        },
      });
    } catch (error) {
      console.error("Private prediction error:", error);
      const message = error instanceof Error ? error.message.trim() : '';
      setPrivateSubmissionError(
        message && !/failed to fetch|networkerror|load failed/i.test(message)
          ? message
          : 'Unable to generate the private price reference. Please try again.',
      );
    }
  };

  if (pricingAccessStatus === 'checking') {
    return (
      <div>
        <PageHeader
          title="Healthcare Cost & Pricing"
          description="Checking system availability."
          gradient
        />
        <div className="container py-16">
          <Card className="max-w-xl mx-auto">
            <CardContent className="p-8 text-center">
              <div className="h-7 w-7 mx-auto mb-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <p className="text-sm text-muted-foreground">
                Checking pricing access...
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (pricingAccessStatus === 'redirecting') {
    return (
      <div>
        <PageHeader
          title="Sign In Required"
          description="Redirecting you to the login page."
          gradient
        />
      </div>
    );
  }

  if (pricingAccessStatus === 'maintenance') {
    return (
      <div>
        <PageHeader
          title={`${systemSettings?.system_name || 'MyCareCost'} Maintenance`}
          description="The healthcare pricing tools are temporarily unavailable."
          gradient
        />
        <div className="container py-16">
          <Card className="max-w-xl mx-auto">
            <CardContent className="p-8 text-center space-y-3">
              <Info className="h-10 w-10 mx-auto text-primary" />
              <h2 className="text-xl font-semibold">
                System maintenance is in progress
              </h2>
              <p className="text-sm text-muted-foreground">
                Pricing and prediction features are temporarily disabled.
                Please try again later.
              </p>
              {systemSettings?.contact_email && (
                <p className="text-sm text-muted-foreground">
                  Contact: {systemSettings.contact_email}
                </p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (pricingAccessStatus === 'error') {
    return (
      <div>
        <PageHeader
          title="Pricing Tools Unavailable"
          description="System availability could not be verified."
          gradient
        />
        <div className="container py-16">
          <Card className="max-w-xl mx-auto">
            <CardContent className="p-8 text-center space-y-3">
              <Info className="h-10 w-10 mx-auto text-destructive" />
              <h2 className="text-xl font-semibold">
                Unable to verify system access
              </h2>
              <p className="text-sm text-muted-foreground">
                {pricingAccessError}
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Healthcare Cost & Pricing" description="Use the NSS 80 primary research model, the isolated US benchmark, or source-supported Malaysian references." gradient />

      <div className="container py-8 md:py-12">
        <Tabs value={tab} onValueChange={setTab} className="max-w-7xl mx-auto">
          {/* Tab Switcher */}
          <TabsList className="grid h-auto min-h-14 w-full grid-cols-2 rounded-xl border border-primary/15 bg-accent p-1.5 mb-10 md:min-h-16 md:grid-cols-4">
            <TabsTrigger
              value="public"
              className="min-h-12 gap-2.5 rounded-lg py-2 text-sm font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md transition-all"
            >
              <Building2 className="h-5 w-5" />
              Public Hospital
            </TabsTrigger>
            <TabsTrigger
              value="private"
              className="min-h-12 gap-2.5 rounded-lg py-2 text-sm font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md transition-all"
            >
              <Building className="h-5 w-5" />
              Private Hospital
            </TabsTrigger>
            <TabsTrigger
              value="research"
              className="min-h-12 gap-2.5 whitespace-normal rounded-lg py-2 text-center text-sm font-semibold leading-tight data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md transition-all"
            >
              <FlaskConical className="h-5 w-5" />
              Prediction Cost / Individual Cost
            </TabsTrigger>
            <TabsTrigger
              value="liam"
              className="min-h-12 gap-2.5 rounded-lg py-2 text-sm font-semibold data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md transition-all"
            >
              <BookOpen className="h-5 w-5" />
              Others
            </TabsTrigger>
          </TabsList>

          {/* ── PUBLIC HOSPITAL ── */}
          <TabsContent value="public">
            <div className="grid gap-6 lg:grid-cols-8">
              {/* Main form */}
              <Card className="min-w-0 border-0 shadow-lg overflow-hidden lg:col-span-3">
                {/* Card header band */}
                <div className="healthcare-gradient px-6 py-5 md:px-8">
                  <h2 className="text-lg font-bold text-primary-foreground flex items-center gap-2">
                    <Calculator className="h-5 w-5" /> Public Hospital Published Pricing
                  </h2>
                  <p className="text-primary-foreground/75 text-sm mt-1">
                    Compare five official hospital datasets with separate MOH national references.
                  </p>
                </div>

                <CardContent className="p-6 md:p-8 space-y-6">
                  {/* Group 1: Treatment */}
                  <fieldset className="space-y-3">
                    <legend className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Treatment Details</legend>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-2 text-sm font-medium">
                        <Stethoscope className="h-4 w-4 text-primary" /> Treatment Category
                      </Label>
                      <Select 
                        value={pubCategory} 
                         disabled={publicCategoriesLoading || publicCategories.length === 0}
                         onValueChange={(v) => {
                           setPubCategory(v);
                           setPubServiceFilter(null);
                            void fetchPublicMatches(v, pubCitizen);
                         }}
                      >
                        <SelectTrigger className="h-11">
                          <SelectValue
                            placeholder={
                              publicCategoriesLoading
                                ? 'Loading treatment categories...'
                                : 'Select treatment category'
                            }
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {publicCategories.map(c => <SelectItem key={c} value={c}>{c}</SelectItem>)}
                        </SelectContent>
                      </Select>
                       {publicCategoriesError && (
                         <p className="text-xs text-destructive pl-1">
                           {publicCategoriesError}
                         </p>
                       )}
                       {pubServiceFilter && (
                         <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-primary/20 bg-primary/5 px-3 py-2 text-sm">
                           <span>
                             Selected assistant service:{' '}
                             <strong>{pubServiceFilter.service}</strong>
                             {pubServiceFilter.mapped_service.toLocaleLowerCase() !==
                               pubServiceFilter.service.toLocaleLowerCase() && (
                               <> ({pubServiceFilter.mapped_service})</>
                             )}
                           </span>
                           <Button
                             type="button"
                             variant="ghost"
                             size="sm"
                             className="h-7 px-2"
                             onClick={clearAssistantServiceFilter}
                           >
                             Clear assistant selection
                           </Button>
                         </div>
                       )}
                     </div>
                  </fieldset>

                  <hr className="border-border" />

                  {/* Group 2: Citizenship */}
                  <fieldset className="space-y-4">
                    <legend className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Patient Information</legend>
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-2">
                        <Label className="flex items-center gap-2 text-sm font-medium">
                          <Users className="h-4 w-4 text-primary" /> Citizenship
                        </Label>
                        <Select 
                          value={pubCitizen}
                           onValueChange={(v) => {
                             setPubCitizen(v);
                             void fetchPublicMatches(
                               pubCategory,
                               v,
                               pubServiceFilter?.service_name,
                             );
                           }}
                        >
                          <SelectTrigger className="h-11"><SelectValue placeholder="Select citizenship" /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="Malaysian">Malaysian Citizen</SelectItem>
                            <SelectItem value="Non-Malaysian">Non-Malaysian</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label className="flex items-center gap-2 text-sm font-medium">
                          <Building2 className="h-4 w-4 text-primary" /> Hospital
                        </Label>
                        <Select
                          value={pubHospital}
                          disabled={publicHospitalsLoading}
                          onValueChange={(value) => {
                            setPubHospital(value);
                            void fetchPublicMatches(
                              pubCategory,
                              pubCitizen,
                              pubServiceFilter?.service_name,
                              value,
                            );
                          }}
                        >
                          <SelectTrigger className="h-11">
                            <SelectValue placeholder="Select hospital" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="all">All Public Hospitals</SelectItem>
                            {publicHospitals.map((hospital) => (
                              <SelectItem
                                key={hospital.hospital_code}
                                value={hospital.hospital_code}
                              >
                                {hospital.hospital_name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </fieldset>

                  {/* Info note */}
                  <div className="flex gap-3 items-start rounded-lg bg-accent/50 border border-accent p-4 text-sm text-accent-foreground">
                    <Info className="h-4 w-4 mt-0.5 shrink-0" />
                    <span>Hospital selection filters published source records. State is displayed from the selected hospital and never used to adjust a charge.</span>
                  </div>

                  {publicSubmissionError && (
                    <p className="text-sm text-destructive" role="alert">
                      {publicSubmissionError}
                    </p>
                  )}

                  <Button className="w-full h-12 text-base font-semibold gap-2" onClick={handlePublicSubmit}
                    disabled={!pubCategory || !pubCitizen}>
                    View Price Reference <ChevronRight className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>

              {/* Matched charges */}
              <div className="min-w-0 lg:col-span-5">
                <Card className="border border-border sticky top-24">
                  <CardContent className="p-5 space-y-4">
                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Matching Published Charges</h3>

                    {pubEstimateLoading ? (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        <div className="h-6 w-6 mx-auto mb-3 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                        Searching charges...
                      </div>
                    ) : pubCategory && pubCitizen ? (
                      <div className="space-y-4 text-sm">
                        <div className="rounded-lg bg-muted/60 px-3 py-2">
                          <p className="font-semibold text-foreground">{pubCategory}</p>
                          <p className="text-xs text-muted-foreground">
                            {pubCitizen} · {publicHospitalSourcesChecked} hospital source{publicHospitalSourcesChecked === 1 ? '' : 's'} checked
                          </p>
                        </div>

                        {pubHospitals.length > 0 ? (
                          <div className="grid min-w-0 items-stretch gap-4 md:grid-cols-2">
                            {hospitalPagination.visibleItems.map((hospital) => {
                              const statistics = hospital.statistics;
                              return (
                            <section
                              key={hospital.hospital_code}
                              className={cn(
                                'flex h-full w-full min-w-0 flex-col overflow-hidden rounded-xl border border-border bg-card',
                                hospitalPagination.visibleItems.length === 1 && 'md:col-span-2',
                              )}
                            >
                              <div className="border-b border-border bg-muted/40 px-4 py-3">
                                <div className="flex items-start gap-3">
                                  <span className="rounded-md bg-primary px-2 py-1 text-xs font-bold text-primary-foreground">
                                    {hospital.hospital_code}
                                  </span>
                                  <div className="min-w-0">
                                    <h4 className="font-semibold leading-tight text-foreground">
                                      {hospital.hospital_name}
                                    </h4>
                                    <p className="mt-1 text-xs text-muted-foreground">
                                      {hospital.state} · Official Published Charges
                                    </p>
                                  </div>
                                </div>
                              </div>

                              {hospital.records.length > 0 ? (
                                <>
                                  <div className="max-h-72 flex-1 divide-y divide-border overflow-y-auto">
                                    {hospital.records.map((charge, idx) => (
                                      <div
                                        key={`${hospital.hospital_code}-${charge.service_name}-${charge.patient_class}-${charge.price_unit || idx}`}
                                        className="space-y-1.5 px-4 py-3"
                                      >
                                        <div className="flex items-start justify-between gap-3">
                                          <p className="min-w-0 flex-1 font-medium leading-snug text-foreground">
                                            {charge.service_name}
                                          </p>
                                          <p className="shrink-0 whitespace-nowrap text-right text-base font-bold text-primary">
                                            {formatPublishedPrice(charge.price_rm)}
                                            {charge.price_unit && (
                                              <span className="ml-1 text-xs font-medium text-muted-foreground">
                                                / {charge.price_unit.replace(/^per\s+/i, '')}
                                              </span>
                                            )}
                                          </p>
                                        </div>
                                        <p className="text-xs capitalize text-muted-foreground">
                                          {charge.ward_class || charge.charge_type}
                                          {charge.patient_class !== 'all'
                                            ? ` · ${charge.patient_class}`
                                            : ''}
                                        </p>
                                        {charge.notes && (
                                          <p className="text-xs text-muted-foreground">
                                            {charge.notes}
                                          </p>
                                        )}
                                      </div>
                                    ))}
                                  </div>

                                  <div className="border-t border-border bg-primary/5 px-4 py-3">
                                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                                      Official Price Summary
                                    </p>
                                    <p className="mt-1 text-xs text-muted-foreground">
                                      Published records: {statistics.records_used}
                                    </p>
                                    {statistics.pricing_type === 'exact' ? (
                                      <p className="mt-2 font-semibold text-foreground">
                                        Published charge: {formatPublishedPrice(statistics.published_cost ?? 0)}
                                      </p>
                                    ) : statistics.pricing_type === 'range' ? (
                                      <div className="mt-2 grid grid-cols-3 gap-2 text-center">
                                        <div>
                                          <p className="text-[11px] text-muted-foreground">Lower</p>
                                          <p className="font-semibold">{formatPublishedPrice(statistics.lower_estimate ?? 0)}</p>
                                        </div>
                                        <div>
                                          <p className="text-[11px] text-muted-foreground">Typical</p>
                                          <p className="font-semibold">{formatPublishedPrice(statistics.typical_estimate ?? 0)}</p>
                                        </div>
                                        <div>
                                          <p className="text-[11px] text-muted-foreground">Upper</p>
                                          <p className="font-semibold">{formatPublishedPrice(statistics.upper_estimate ?? 0)}</p>
                                        </div>
                                      </div>
                                    ) : null}
                                  </div>
                                </>
                              ) : (
                                <div className="flex flex-1 items-center justify-center px-4 py-6 text-center text-xs leading-relaxed text-muted-foreground">
                                  No matching published charge was found for this treatment category in the current {hospital.hospital_code} dataset.
                                </div>
                              )}

                              <div className="mt-auto border-t border-border px-4 py-3">
                                <p className="mb-2 text-xs text-muted-foreground">
                                  Source: {hospital.source_name}
                                </p>
                                <a
                                  href={hospital.source_url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-primary hover:underline"
                                >
                                  View Official Source
                                  <ExternalLink className="h-3 w-3" />
                                </a>
                              </div>
                            </section>
                              );
                            })}
                          </div>
                        ) : (
                          <div className="rounded-xl border border-dashed border-border px-4 py-8 text-center text-sm text-muted-foreground">
                            No matching hospital-specific published charge was found for this selection.
                          </div>
                        )}

                        {hospitalPagination.pageCount > 1 && (
                          <nav
                            aria-label="Hospital result pages"
                            className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted/30 px-3 py-3"
                          >
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="order-2 flex-1 gap-1.5 sm:order-none sm:flex-none"
                              disabled={!hospitalPagination.hasPreviousPage}
                              onClick={() => setCurrentHospitalPage(
                                hospitalPagination.currentPage - 1,
                              )}
                              aria-label="Show previous hospital results page"
                            >
                              <ChevronLeft className="h-4 w-4" />
                              Previous
                            </Button>

                            <div
                              className="order-1 w-full text-center sm:order-none sm:w-auto"
                              role="status"
                              aria-live="polite"
                            >
                              <p className="font-semibold text-foreground">
                                Page {hospitalPagination.currentPage} of {hospitalPagination.pageCount}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                Showing {hospitalPagination.showingStart}
                                {hospitalPagination.showingStart === hospitalPagination.showingEnd
                                  ? ''
                                  : `–${hospitalPagination.showingEnd}`}{' '}
                                of {pubHospitals.length} hospitals
                              </p>
                            </div>

                            <Button
                              type="button"
                              size="sm"
                              className="order-3 flex-1 gap-1.5 sm:order-none sm:flex-none"
                              disabled={!hospitalPagination.hasNextPage}
                              onClick={() => setCurrentHospitalPage(
                                hospitalPagination.currentPage + 1,
                              )}
                              aria-label="Show next hospital results page"
                            >
                              Next Page
                              <ChevronRight className="h-4 w-4" />
                            </Button>
                          </nav>
                        )}

                        {pubNationalReferences.length > 0 && (
                          <section className="overflow-hidden rounded-xl border border-primary/25 bg-primary/5">
                            <div className="border-b border-primary/20 px-4 py-3">
                              <p className="text-xs font-bold uppercase tracking-wider text-primary">
                                MOH National Reference
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                National published schedules are shown separately and are not hospitals.
                              </p>
                            </div>
                            <div className="grid gap-4 p-4 xl:grid-cols-2">
                              {pubNationalReferences.map((reference) => (
                                <article
                                  key={reference.source_code}
                                  className={cn(
                                    'w-full min-w-0 overflow-hidden rounded-lg border border-border bg-card',
                                    pubNationalReferences.length === 1 && 'xl:col-span-2',
                                  )}
                                >
                                  <div className="border-b border-border px-3 py-2.5">
                                    <h4 className="font-semibold text-foreground">{reference.source_name}</h4>
                                    {reference.source_updated_at && (
                                      <p className="mt-1 text-xs text-muted-foreground">
                                        Source updated: {reference.source_updated_at}
                                      </p>
                                    )}
                                  </div>
                                  <div className="max-h-72 divide-y divide-border overflow-y-auto">
                                    {reference.records.map((charge, index) => (
                                      <div key={charge.id || `${reference.source_code}-${index}`} className="flex items-start justify-between gap-3 px-3 py-2.5">
                                        <div className="min-w-0 flex-1">
                                          <p className="font-medium leading-snug text-foreground">{charge.service_name}</p>
                                          <p className="mt-1 text-xs text-muted-foreground">
                                            {charge.ward_class || charge.patient_type || charge.charge_type}
                                          </p>
                                        </div>
                                        <p className="shrink-0 whitespace-nowrap text-right font-bold text-primary">
                                          {formatPublishedPrice(charge.price_rm)}
                                          {charge.price_unit && (
                                            <span className="ml-1 text-xs font-medium text-muted-foreground">
                                              / {charge.price_unit.replace(/^per\s+/i, '')}
                                            </span>
                                          )}
                                        </p>
                                      </div>
                                    ))}
                                  </div>
                                  <div className="border-t border-border px-3 py-2.5">
                                    <a href={reference.source_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1.5 text-xs font-semibold text-primary hover:underline">
                                      View Official Source <ExternalLink className="h-3 w-3" />
                                    </a>
                                  </div>
                                </article>
                              ))}
                            </div>
                          </section>
                        )}

                        <p className="rounded-lg border border-border bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground">
                          These are published reference charges, not guaranteed final bills or medical advice. Actual charges may vary by ward class, treatment requirements, patient eligibility, additional procedures, and current hospital policies. Confirm current fees with the hospital.
                        </p>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        <Building2 className="h-8 w-8 mx-auto mb-3 opacity-40" />
                        Select a treatment category and citizenship to compare official hospital charges.
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* ── PRIVATE HOSPITAL ── */}
          <TabsContent value="private">
            <div className="grid gap-6 lg:grid-cols-5">
              {/* Main form – 3 cols */}
              <Card className="border-0 shadow-lg overflow-hidden lg:col-span-3">
                <div className="healthcare-gradient px-6 py-5 md:px-8">
                  <h2 className="text-lg font-bold text-primary-foreground flex items-center gap-2">
                    <Calculator className="h-5 w-5" /> Private Hospital Package Reference
                  </h2>
                  <p className="text-primary-foreground/75 text-sm mt-1">
                    Published package price plus published ward price for the selected stay.
                  </p>
                </div>

                <CardContent className="p-6 md:p-8 space-y-6">
                  {/* Group: Package */}
                  <fieldset className="space-y-3">
                    <legend className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Private Hospital</legend>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-2 text-sm font-medium">
                        <Building className="h-4 w-4 text-primary" /> Select Hospital
                      </Label>
                      <Select
                        value={privHospital}
                        onValueChange={(value) => {
                          setPrivHospital(value);
                          setPrivPackage('');
                        }}
                      >
                        <SelectTrigger className="h-11">
                          <SelectValue
                            placeholder={
                              privateOptionsLoading && privateHospitals.length === 0
                                ? 'Loading private hospitals...'
                                : 'Choose private hospital'
                            }
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {privateHospitals.map((hospital) => (
                            <SelectItem key={hospital} value={hospital}>
                              {hospital}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {privateOptionsError && !privHospital && (
                        <p className="text-xs text-destructive pl-1">
                          {privateOptionsError}
                        </p>
                      )}
                    </div>
                  </fieldset>

                  <hr className="border-border" />

                  {/* Group: Package */}
                  <fieldset className="space-y-3">
                    <legend className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Hospital Package</legend>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-2 text-sm font-medium">
                        <Package className="h-4 w-4 text-primary" /> Select Package
                      </Label>
                      <Select value={privPackage} onValueChange={setPrivPackage}>
                        <SelectTrigger className="h-11">
                          <SelectValue
                            placeholder={
                              !privHospital
                                ? 'Choose hospital first'
                                : privateOptionsLoading
                                  ? 'Loading hospital packages...'
                                  : 'Choose a hospital package'
                            }
                          />
                        </SelectTrigger>
                        <SelectContent>
                          {hospitalPackages.map(p => (
                            <SelectItem key={p.id} value={p.id}>
                              {p.name} — RM {p.price.toLocaleString()}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {selectedPkg && (
                        <p className="text-xs text-muted-foreground pl-1">
                          {selectedPkg.description}
                        </p>
                      )}
                      {privateOptionsError && privHospital && (
                        <p className="text-xs text-destructive pl-1">
                          {privateOptionsError}
                        </p>
                      )}
                    </div>
                  </fieldset>

                  <hr className="border-border" />

                  {/* Group: Ward & Stay */}
                  <fieldset className="space-y-4">
                    <legend className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">Ward & Stay</legend>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-2 text-sm font-medium">
                        <BedDouble className="h-4 w-4 text-primary" /> Ward Type
                      </Label>
                      <Select
                        value={privWard}
                        disabled={!privHospital || wardRatesLoading || hospitalWardRates.length === 0}
                        onValueChange={(ward) => {
                          setPrivWard(ward);
                          setPrivNights(ward === 'No Stay' ? '0' : (privNights === '0' ? '1' : privNights));
                        }}
                      >
                        <SelectTrigger className="h-11">
                          <SelectValue placeholder={
                            !privHospital
                              ? 'Choose hospital first'
                              : wardRatesLoading
                                ? 'Loading ward rates...'
                                : 'Choose ward type'
                          } />
                        </SelectTrigger>
                        <SelectContent>
                          {hospitalWardRates.map(w => (
                            <SelectItem key={w.id} value={w.name}>
                              {w.name}{w.name === 'No Stay' ? ' — No overnight admission' : ` — RM ${w.dailyRate}/night`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {wardRatesError && (
                        <p className="text-xs text-destructive pl-1">{wardRatesError}</p>
                      )}
                    </div>
                    <div className="space-y-2">
                      <Label className="flex items-center gap-2 text-sm font-medium">
                        <CalendarDays className="h-4 w-4 text-primary" /> Estimated Stay (nights)
                      </Label>
                      <Input
                        type="number"
                        min={hasNoStay ? '0' : '1'}
                        max="30"
                        value={privNights}
                        onChange={e => setPrivNights(e.target.value)}
                        disabled={hasNoStay}
                        className="h-11"
                      />
                    </div>
                  </fieldset>

                  {privateSubmissionError && (
                    <p className="text-sm text-destructive" role="alert">
                      {privateSubmissionError}
                    </p>
                  )}

                  <Button className="w-full h-12 text-base font-semibold gap-2" onClick={handlePrivateSubmit}
                    disabled={!privHospital || !privPackage || !privWard || !privNights}>
                    View Price Reference <ChevronRight className="h-4 w-4" />
                  </Button>
                </CardContent>
              </Card>

              {/* Live summary sidebar – 2 cols */}
              <div className="lg:col-span-2">
                <Card className="border border-border sticky top-24">
                  <CardContent className="p-5 space-y-4">
                    <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">Published Price Summary</h3>

                    {liveCost ? (
                      <>
                        <div className="space-y-3 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Published Package Price</span>
                            <span className="font-medium">RM {liveCost.packageCost.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">
                              {hasNoStay ? 'Published Ward Price (No Stay)' : `Published Ward Price (${privNights} night${parseInt(privNights) !== 1 ? 's' : ''})`}
                            </span>
                            <span className="font-medium">RM {liveCost.wardCost.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Surgeon Fee</span>
                            <span className="font-medium text-right">Not included in published pricing data</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Misc & Admin</span>
                            <span className="font-medium text-right">Not included in published pricing data</span>
                          </div>
                        </div>
                        <hr className="border-border" />
                        <div className="flex justify-between items-baseline">
                          <span className="text-sm font-semibold">Published Reference Total</span>
                          <span className="text-2xl font-bold text-primary">RM {liveCost.total.toLocaleString()}</span>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground text-sm">
                        <Calculator className="h-8 w-8 mx-auto mb-3 opacity-40" />
                        Select a package and ward to see the published price reference.
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="research">
            <AIPredictionForm />
          </TabsContent>

          <TabsContent value="liam">
            <LIAMPricingReference />
          </TabsContent>
        </Tabs>
        <HealthcareServiceAssistant
          onUseService={handleUseAssistantService}
          availableCategories={publicCategories}
          categoriesLoading={publicCategoriesLoading}
          categoriesError={publicCategoriesError}
        />
      </div>
    </div>
  );
};

export default Predict;
