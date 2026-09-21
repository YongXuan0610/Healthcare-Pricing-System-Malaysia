export type PublicPatientClass =
  | 'all'
  | 'citizen'
  | 'foreigner'
  | 'unhcr'
  | 'resident';

export interface PublicHospitalOption {
  hospital_code: string;
  hospital_name: string;
  state: string;
  source_url: string;
  source_name: string;
}

export interface PublicCharge {
  hospital_name?: string;
  id?: string;
  source_type?: 'hospital' | 'national_reference';
  source_code?: string;
  hospital_code?: string;
  category: string;
  service_name: string;
  original_service_name?: string;
  patient_class: PublicPatientClass;
  citizenship?: string;
  patient_type?: string;
  charge_type: string;
  ward_class?: string;
  room_type?: string;
  price_rm: number;
  price_unit?: string;
  notes?: string;
  source_url: string;
  source_name?: string;
  original_price_text?: string;
  scraped_at?: string;
  source_updated_at?: string;
}

export interface PublicPriceStatistics {
  estimate_available: boolean;
  pricing_type: 'unavailable' | 'exact' | 'range';
  published_cost: number | null;
  lower_estimate: number | null;
  typical_estimate: number | null;
  upper_estimate: number | null;
  records_used: number;
  range_method:
    | 'single_published_charge'
    | 'min_max'
    | 'p25_median_p75'
    | null;
}

export interface PublicHospitalPricing {
  hospital_code: string;
  hospital_name: string;
  state: string;
  source_url: string;
  source_name: string;
  records: PublicCharge[];
  statistics: PublicPriceStatistics;
}

export interface PublicNationalReference {
  source_code: string;
  source_name: string;
  state: 'National' | string;
  source_url: string;
  source_updated_at?: string;
  records: PublicCharge[];
}

export interface PublicPricingApiResponse extends PublicPriceStatistics {
  type: 'public';
  predicted_cost: number | null;
  currency: 'MYR';
  pricing_source: 'official_public_dataset' | 'unavailable';
  pricing_scope: string;
  methodology: string;
  message: string | null;
  hospitals: PublicHospitalPricing[];
  national_references: PublicNationalReference[];
  hospital_filter: string;
  hospital_sources_checked: number;
  matched_public_charges: PublicCharge[];
}

export interface PricingBreakdownItem {
  label: string;
  amount: number | null;
  percentage: number | null;
}

interface PublicPricingResultBase {
  type: 'public';
  date: string;
  currency: 'MYR';
  inputs: Record<string, string>;
  breakdown: PricingBreakdownItem[];
  pricing_source: 'official_public_dataset' | 'unavailable';
  pricing_scope: string;
  methodology: string;
  matched_public_charges: PublicCharge[];
  hospitals?: PublicHospitalPricing[];
  national_references?: PublicNationalReference[];
}

interface UnavailablePublicPricingResult extends PublicPricingResultBase {
  estimate_available: false;
  pricing_type: 'unavailable';
  totalCost: null;
  published_cost: null;
  lower_estimate: null;
  typical_estimate: null;
  upper_estimate: null;
  records_used: 0;
  range_method: null;
  message: string;
}

interface ExactPublicPricingResult extends PublicPricingResultBase {
  estimate_available: true;
  pricing_type: 'exact';
  totalCost: number;
  published_cost: number;
  lower_estimate: null;
  typical_estimate: number;
  upper_estimate: null;
  records_used: 1;
  range_method: 'single_published_charge';
  message: null;
}

interface RangePublicPricingResult extends PublicPricingResultBase {
  estimate_available: true;
  pricing_type: 'range';
  totalCost: number;
  published_cost: null;
  lower_estimate: number;
  typical_estimate: number;
  upper_estimate: number;
  records_used: number;
  range_method: 'min_max' | 'p25_median_p75';
  message: null;
}

export type PublicPricingResult =
  | UnavailablePublicPricingResult
  | ExactPublicPricingResult
  | RangePublicPricingResult;

export interface PrivatePricingResult {
  type: 'private';
  date: string;
  totalCost: number;
  inputs: Record<string, string>;
  breakdown: PricingBreakdownItem[];
  message?: string;
}

export type HospitalPricingResult = PublicPricingResult | PrivatePricingResult;
