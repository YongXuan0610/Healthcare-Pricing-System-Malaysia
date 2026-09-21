import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  CircleHelp,
  Loader2,
  MessageCircle,
  RotateCcw,
  Search,
  ShieldAlert,
  X,
} from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  recommendHealthcareService,
  type HealthcareServiceAssistantResponse,
  type HealthcareServiceSuggestion,
} from '@/lib/api';
import { getAvailableHealthcareServiceGroups } from '@/lib/healthcareServiceGroups';

type HealthcareServiceAssistantProps = {
  onUseService: (suggestion: HealthcareServiceSuggestion) => void;
  availableCategories: string[];
  categoriesLoading?: boolean;
  categoriesError?: string;
};

type AssistantStep = 'groups' | 'services' | 'recommendation' | 'fallback';
type RecommendationSource = 'guided' | 'fallback';

const MAX_MESSAGE_LENGTH = 500;

const HealthcareServiceAssistant = ({
  onUseService,
  availableCategories,
  categoriesLoading = false,
  categoriesError = '',
}: HealthcareServiceAssistantProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [assistantStep, setAssistantStep] = useState<AssistantStep>('groups');
  const [selectedServiceGroup, setSelectedServiceGroup] = useState<string | null>(
    null,
  );
  const [selectedServiceCategory, setSelectedServiceCategory] = useState<
    string | null
  >(null);
  const [selectedSuggestion, setSelectedSuggestion] =
    useState<HealthcareServiceSuggestion | null>(null);
  const [recommendationSource, setRecommendationSource] =
    useState<RecommendationSource>('guided');
  const [description, setDescription] = useState('');
  const [result, setResult] =
    useState<HealthcareServiceAssistantResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const availableGroups = useMemo(
    () => getAvailableHealthcareServiceGroups(availableCategories),
    [availableCategories],
  );
  const selectedGroup = availableGroups.find(
    (group) => group.id === selectedServiceGroup,
  );

  useEffect(() => {
    if (!isOpen) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  const clearTransientState = () => {
    setSelectedServiceCategory(null);
    setSelectedSuggestion(null);
    setDescription('');
    setResult(null);
    setError('');
  };

  const handleStartOver = () => {
    setAssistantStep('groups');
    setSelectedServiceGroup(null);
    clearTransientState();
  };

  const handleSelectGroup = (groupId: string) => {
    setSelectedServiceGroup(groupId);
    setSelectedServiceCategory(null);
    setSelectedSuggestion(null);
    setResult(null);
    setError('');
    setAssistantStep('services');
  };

  const showRecommendation = (
    suggestion: HealthcareServiceSuggestion,
    source: RecommendationSource,
  ) => {
    const availableCategory = availableCategories.find(
      (category) =>
        category.toLocaleLowerCase() === suggestion.category.toLocaleLowerCase(),
    );

    if (!availableCategory) {
      setError(
        'That pricing category is not currently available. Please choose one of the guided service categories.',
      );
      setResult(null);
      setAssistantStep(source === 'fallback' ? 'fallback' : 'groups');
      return;
    }

    setSelectedServiceCategory(availableCategory);
    setSelectedSuggestion({
      ...suggestion,
      category: availableCategory,
    });
    setRecommendationSource(source);
    setError('');
    setAssistantStep('recommendation');
  };

  const handleSelectCategory = (category: string) => {
    showRecommendation(
      {
        service: category,
        mapped_service: category,
        service_name: null,
        category,
        confidence: 1,
        message: `Official published ${category.toLocaleLowerCase()} charges are available in the Public Hospital pricing section.`,
      },
      'guided',
    );
  };

  const handleFallbackSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedDescription = description.trim();

    if (!trimmedDescription) {
      setError('Please describe the type of healthcare service you want to find.');
      setResult(null);
      return;
    }

    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      const recommendation = await recommendHealthcareService(trimmedDescription);

      if (
        recommendation.status === 'matched' &&
        recommendation.service &&
        recommendation.mapped_service &&
        recommendation.category &&
        recommendation.confidence !== null
      ) {
        showRecommendation(
          {
            service: recommendation.service,
            mapped_service: recommendation.mapped_service,
            service_name: recommendation.service_name,
            category: recommendation.category,
            confidence: recommendation.confidence,
            message: recommendation.message,
          },
          'fallback',
        );
      } else {
        setResult(recommendation);
      }
    } catch (requestError) {
      const message =
        requestError instanceof Error ? requestError.message.trim() : '';
      setError(
        message && !/failed to fetch|networkerror|load failed/i.test(message)
          ? message
          : 'The Healthcare Service Assistant is temporarily unavailable. Please select a category from the guided list.',
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleTryAgain = () => {
    setDescription('');
    setResult(null);
    setError('');
  };

  const handleRecommendationBack = () => {
    setSelectedServiceCategory(null);
    setSelectedSuggestion(null);
    setError('');
    setAssistantStep(
      recommendationSource === 'fallback' ? 'fallback' : 'services',
    );
  };

  const handleUseService = () => {
    if (!selectedSuggestion) return;
    onUseService(selectedSuggestion);
    setIsOpen(false);
  };

  return (
    <>
      <Button
        type="button"
        size="icon"
        className="fixed bottom-4 right-4 z-[60] h-14 w-14 rounded-full shadow-xl transition-transform hover:scale-105 sm:bottom-6 sm:right-6"
        aria-label={
          isOpen
            ? 'Minimise Healthcare Service Assistant'
            : 'Open Healthcare Service Assistant'
        }
        aria-controls="healthcare-service-assistant-panel"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((open) => !open)}
      >
        <MessageCircle className="h-6 w-6" />
      </Button>

      {isOpen && (
        <section
          id="healthcare-service-assistant-panel"
          role="dialog"
          aria-labelledby="healthcare-service-assistant-title"
          className="fixed bottom-20 left-4 right-4 z-50 flex w-auto max-w-[400px] flex-col overflow-hidden rounded-2xl border border-border bg-card text-card-foreground shadow-2xl animate-in fade-in slide-in-from-bottom-4 sm:bottom-24 sm:left-auto sm:right-6 sm:w-[400px]"
          style={{ maxHeight: 'min(640px, calc(100vh - 7rem))' }}
        >
          <header className="flex shrink-0 items-start gap-3 bg-primary px-4 py-4 text-primary-foreground">
            <div className="rounded-lg bg-primary-foreground/15 p-2">
              <MessageCircle className="h-5 w-5" />
            </div>
            <div className="min-w-0 flex-1">
              <h2
                id="healthcare-service-assistant-title"
                className="font-semibold leading-tight"
              >
                Healthcare Service Assistant
              </h2>
              <p className="mt-1 text-xs text-primary-foreground/80">
                Find the right healthcare pricing category
              </p>
            </div>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-9 w-9 text-primary-foreground hover:bg-primary-foreground/15 hover:text-primary-foreground"
              aria-label="Close Healthcare Service Assistant"
              onClick={() => setIsOpen(false)}
            >
              <X className="h-5 w-5" />
            </Button>
          </header>

          <div className="flex flex-col gap-5 overflow-y-auto p-4 sm:p-5">
            <p className="rounded-lg bg-muted/60 p-3 text-sm text-muted-foreground">
              Choose the type of healthcare service you want to look up. This
              assistant helps locate pricing categories; it does not diagnose
              conditions or provide medical advice.
            </p>

            {assistantStep === 'groups' && (
              <div className="space-y-4">
                <div>
                  <p className="font-semibold">
                    What type of healthcare service are you looking for?
                  </p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Choose a broad service group to see available pricing
                    categories.
                  </p>
                </div>

                {categoriesLoading ? (
                  <div
                    className="flex items-center gap-2 rounded-lg border border-border p-3 text-sm text-muted-foreground"
                    role="status"
                  >
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading available service categories...
                  </div>
                ) : availableGroups.length > 0 ? (
                  <div className="grid grid-cols-2 gap-2">
                    {availableGroups.map((group) => (
                      <Button
                        key={group.id}
                        type="button"
                        variant="outline"
                        className="h-auto min-h-14 justify-between gap-2 whitespace-normal px-3 py-3 text-left leading-snug hover:border-primary/50 hover:bg-primary/5"
                        onClick={() => handleSelectGroup(group.id)}
                      >
                        <span>{group.label}</span>
                        <ArrowRight className="h-4 w-4 shrink-0 text-primary" />
                      </Button>
                    ))}
                  </div>
                ) : (
                  <p className="rounded-lg border border-border p-3 text-sm text-muted-foreground">
                    No guided pricing categories are currently available.
                  </p>
                )}

                {(categoriesError || error) && (
                  <p className="text-sm text-destructive" role="alert">
                    {error || categoriesError}
                  </p>
                )}

                <Button
                  type="button"
                  variant="ghost"
                  className="w-full gap-2 border border-dashed border-border"
                  onClick={() => {
                    setError('');
                    setResult(null);
                    setAssistantStep('fallback');
                  }}
                >
                  <CircleHelp className="h-4 w-4" />
                  I&apos;m not sure
                </Button>
              </div>
            )}

            {assistantStep === 'services' && selectedGroup && (
              <div className="space-y-4">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="-ml-2 w-fit"
                  onClick={handleStartOver}
                >
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  Back
                </Button>
                <div className="space-y-2">
                  <Badge variant="secondary">{selectedGroup.label}</Badge>
                  <div>
                    <p className="font-semibold">Which service do you need?</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Select the pricing category you want to view.
                    </p>
                  </div>
                </div>
                <div className="grid gap-2">
                  {selectedGroup.categories.map((category) => (
                    <Button
                      key={category}
                      type="button"
                      variant="outline"
                      className="h-auto min-h-11 justify-between whitespace-normal px-3 py-2 text-left hover:border-primary/50 hover:bg-primary/5"
                      onClick={() => handleSelectCategory(category)}
                    >
                      <span>{category}</span>
                      <ArrowRight className="h-4 w-4 shrink-0 text-primary" />
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {assistantStep === 'recommendation' &&
              selectedServiceCategory &&
              selectedSuggestion && (
                <div className="space-y-4">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="-ml-2 w-fit"
                    onClick={handleRecommendationBack}
                  >
                    <ChevronLeft className="mr-1 h-4 w-4" />
                    Back
                  </Button>
                  <div className="space-y-4 rounded-lg border border-primary/20 bg-primary/5 p-4">
                    <div className="flex items-start gap-3">
                      <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
                      <div className="min-w-0 space-y-2">
                        <p className="text-sm font-semibold text-muted-foreground">
                          Recommended Pricing Category
                        </p>
                        <p className="text-lg font-semibold text-primary">
                          {selectedServiceCategory}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          {selectedSuggestion.message}
                        </p>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      This selection only helps you navigate published pricing
                      information. It is not a diagnosis or medical advice.
                    </p>
                    <Button
                      type="button"
                      className="w-full"
                      onClick={handleUseService}
                    >
                      View Prices
                    </Button>
                  </div>
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full"
                    onClick={handleStartOver}
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Start Over
                  </Button>
                </div>
              )}

            {assistantStep === 'fallback' && (
              <div className="space-y-4">
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="-ml-2 w-fit"
                  onClick={handleStartOver}
                >
                  <ChevronLeft className="mr-1 h-4 w-4" />
                  Back
                </Button>
                <div>
                  <p className="font-semibold">Not sure which category?</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Describe the service or pricing information you are trying
                    to find. This optional matcher uses only supported pricing
                    categories.
                  </p>
                </div>

                <form className="space-y-3" onSubmit={handleFallbackSubmit}>
                  <div className="space-y-2">
                    <Label htmlFor="healthcare-service-description">
                      Describe the type of service you are trying to find
                    </Label>
                    <Textarea
                      id="healthcare-service-description"
                      value={description}
                      maxLength={MAX_MESSAGE_LENGTH}
                      rows={3}
                      placeholder="For example: I am looking for physiotherapy pricing."
                      onChange={(event) => {
                        setDescription(event.target.value);
                        setError('');
                      }}
                    />
                    <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                      <span>Do not use this tool for emergency assessment.</span>
                      <span>{description.length}/{MAX_MESSAGE_LENGTH}</span>
                    </div>
                  </div>

                  {error && (
                    <p className="text-sm text-destructive" role="alert">
                      {error}
                    </p>
                  )}

                  <Button
                    type="submit"
                    className="w-full gap-2"
                    disabled={isLoading || description.trim().length === 0}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Finding a pricing category...
                      </>
                    ) : (
                      <>
                        <Search className="h-4 w-4" />
                        Find Pricing Category
                      </>
                    )}
                  </Button>
                </form>

                {result?.status === 'ambiguous' && (
                  <div className="space-y-4 rounded-lg border border-border bg-muted/30 p-4">
                    <div className="flex items-start gap-3">
                      <CircleHelp className="mt-0.5 h-5 w-5 shrink-0 text-primary" />
                      <div>
                        <p className="font-semibold">Choose a Pricing Category</p>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {result.message}
                        </p>
                      </div>
                    </div>
                    <div className="grid gap-2">
                      {result.suggestions.map((suggestion) => (
                        <Button
                          key={`${suggestion.service}-${suggestion.category}`}
                          type="button"
                          variant="outline"
                          className="h-auto min-h-11 justify-between whitespace-normal px-3 py-2 text-left"
                          onClick={() =>
                            showRecommendation(suggestion, 'fallback')
                          }
                        >
                          <span>{suggestion.category}</span>
                          <ArrowRight className="h-4 w-4 shrink-0 text-primary" />
                        </Button>
                      ))}
                    </div>
                    <Button type="button" variant="ghost" onClick={handleTryAgain}>
                      <RotateCcw className="mr-2 h-4 w-4" />
                      Try Another Description
                    </Button>
                  </div>
                )}

                {result?.status === 'unmatched' && (
                  <div className="space-y-3 rounded-lg border border-border bg-muted/30 p-4">
                    <div className="flex items-start gap-3">
                      <CircleHelp className="mt-0.5 h-5 w-5 shrink-0 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        {result.message}
                      </p>
                    </div>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={handleTryAgain}
                    >
                      <RotateCcw className="mr-2 h-4 w-4" />
                      Try Another Description
                    </Button>
                  </div>
                )}

                {result?.status === 'urgent' && (
                  <div
                    className="flex items-start gap-3 rounded-lg border border-destructive/30 bg-destructive/5 p-4"
                    role="alert"
                  >
                    <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
                    <p className="text-sm">{result.message}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      )}
    </>
  );
};

export default HealthcareServiceAssistant;
