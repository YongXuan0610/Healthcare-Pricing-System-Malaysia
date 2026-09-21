const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = (
  configuredApiBaseUrl || "http://localhost:8000"
).replace(/\/+$/, "");

export const apiUrl = (path: string): string =>
  `${API_BASE_URL}${path.startsWith("/") ? path : `/${path}`}`;

export type HealthcareServiceAssistantStatus =
  | 'matched'
  | 'ambiguous'
  | 'unmatched'
  | 'urgent';

export type HealthcareServiceSuggestion = {
  service: string;
  mapped_service: string;
  service_name: string | null;
  category: string;
  confidence: number;
  message: string;
};

export type HealthcareServiceAssistantResponse = {
  status: HealthcareServiceAssistantStatus;
  service: string | null;
  mapped_service: string | null;
  service_name: string | null;
  category: string | null;
  confidence: number | null;
  message: string;
  suggestions: HealthcareServiceSuggestion[];
};

export const recommendHealthcareService = async (
  message: string,
  signal?: AbortSignal,
): Promise<HealthcareServiceAssistantResponse> => {
  const response = await fetch(apiUrl('/assistant/recommend-service'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
    signal,
  });

  const data: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      data &&
      typeof data === 'object' &&
      'detail' in data &&
      typeof data.detail === 'string'
        ? data.detail
        : 'The Healthcare Service Assistant is temporarily unavailable.';
    throw new Error(detail);
  }

  return data as HealthcareServiceAssistantResponse;
};
