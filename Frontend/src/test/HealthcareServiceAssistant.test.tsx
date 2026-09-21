import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import HealthcareServiceAssistant from '@/components/HealthcareServiceAssistant';
import { recommendHealthcareService } from '@/lib/api';
import { getAvailableHealthcareServiceGroups } from '@/lib/healthcareServiceGroups';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...actual,
    recommendHealthcareService: vi.fn(),
  };
});

const availableCategories = [
  'Audiology',
  'Delivery / Maternity',
  'Dietetics',
  'Emergency',
  'Inpatient Treatment',
  'Laboratory',
  'Medical Reports',
  'Nephrology',
  'Occupational Therapy',
  'Other Published Charges',
  'Outpatient',
  'Physiotherapy',
  'Procedures',
  'Radiology / Imaging',
  'Radiotherapy / Oncology',
  'Specialist Clinic',
  'Speech Therapy',
  'Traditional and Complementary Medicine',
  'Treatment Charges',
  'Ward Charges',
  'Ward Deposit',
];

const openAssistant = () => {
  fireEvent.click(
    screen.getByRole('button', {
      name: 'Open Healthcare Service Assistant',
    }),
  );
};

const renderAssistant = (onUseService = vi.fn()) => {
  render(
    <HealthcareServiceAssistant
      availableCategories={availableCategories}
      onUseService={onUseService}
    />,
  );
  openAssistant();
  return onUseService;
};

describe('HealthcareServiceAssistant', () => {
  beforeEach(() => {
    vi.mocked(recommendHealthcareService).mockReset();
  });

  it('opens with supported broad groups and no primary free-text prompt', () => {
    renderAssistant();

    expect(
      screen.getByText('What type of healthcare service are you looking for?'),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Hospital Stay / Ward' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Rehabilitation / Therapy' }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/does not diagnose conditions or provide medical advice/i),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText('Describe the type of service you are trying to find'),
    ).not.toBeInTheDocument();
  });

  it('selects a guided category and applies it through View Prices', () => {
    const onUseService = renderAssistant();

    fireEvent.click(
      screen.getByRole('button', { name: 'Rehabilitation / Therapy' }),
    );
    expect(screen.getByText('Which service do you need?')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Physiotherapy' })).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Occupational Therapy' }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Physiotherapy' }));
    expect(screen.getByText('Recommended Pricing Category')).toBeInTheDocument();
    expect(screen.getByText('Physiotherapy')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'View Prices' }));
    expect(onUseService).toHaveBeenCalledWith(
      expect.objectContaining({
        service: 'Physiotherapy',
        mapped_service: 'Physiotherapy',
        service_name: null,
        category: 'Physiotherapy',
      }),
    );
  });

  it('supports Back navigation and Start Over without closing the popup', () => {
    renderAssistant();

    fireEvent.click(
      screen.getByRole('button', { name: 'Maternity / Delivery' }),
    );
    fireEvent.click(
      screen.getByRole('button', { name: 'Delivery / Maternity' }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(
      screen.getByRole('button', { name: 'Delivery / Maternity' }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Back' }));
    expect(
      screen.getByRole('button', { name: 'Maternity / Delivery' }),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole('button', { name: 'Tests & Imaging' }),
    );
    fireEvent.click(
      screen.getByRole('button', { name: 'Radiology / Imaging' }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start Over' }));
    expect(
      screen.getByRole('button', { name: 'Tests & Imaging' }),
    ).toBeInTheDocument();
    expect(screen.queryByText('Recommended Pricing Category')).not.toBeInTheDocument();
  });

  it('never exposes mapped categories that are absent from the API list', () => {
    const groups = getAvailableHealthcareServiceGroups([
      'Physiotherapy',
      'Unsupported Category',
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0]).toMatchObject({
      label: 'Rehabilitation / Therapy',
      categories: ['Physiotherapy'],
    });

    const allMappedCategories = getAvailableHealthcareServiceGroups(
      availableCategories,
    ).flatMap((group) => group.categories);
    expect([...allMappedCategories].sort()).toEqual(
      [...availableCategories].sort(),
    );
  });

  it("keeps free text behind I'm not sure and preserves its 500-character limit", async () => {
    vi.mocked(recommendHealthcareService).mockResolvedValue({
      status: 'matched',
      service: 'Physiotherapy',
      mapped_service: 'Physiotherapy',
      service_name: null,
      category: 'Physiotherapy',
      confidence: 0.92,
      message: 'Official published physiotherapy charges are available.',
      suggestions: [],
    });
    const onUseService = renderAssistant();

    fireEvent.click(screen.getByRole('button', { name: "I'm not sure" }));
    const textarea = screen.getByLabelText(
      'Describe the type of service you are trying to find',
    );
    expect(textarea).toHaveAttribute('maxlength', '500');
    fireEvent.change(textarea, {
      target: { value: 'I am looking for physiotherapy pricing.' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Find Pricing Category' }));

    await waitFor(() =>
      expect(recommendHealthcareService).toHaveBeenCalledWith(
        'I am looking for physiotherapy pricing.',
      ),
    );
    expect(await screen.findByText('Recommended Pricing Category')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'View Prices' }));
    expect(onUseService).toHaveBeenCalledWith(
      expect.objectContaining({ category: 'Physiotherapy' }),
    );
  });
});
