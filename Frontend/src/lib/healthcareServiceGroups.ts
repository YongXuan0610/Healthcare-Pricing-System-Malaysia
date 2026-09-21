export type HealthcareServiceGroup = {
  id: string;
  label: string;
  categories: readonly string[];
};

export type AvailableHealthcareServiceGroup = Omit<
  HealthcareServiceGroup,
  'categories'
> & {
  categories: string[];
};

export const HEALTHCARE_SERVICE_GROUPS: readonly HealthcareServiceGroup[] = [
  {
    id: 'hospital-stay',
    label: 'Hospital Stay / Ward',
    categories: ['Ward Charges', 'Inpatient Treatment', 'Ward Deposit'],
  },
  {
    id: 'clinic-consultation',
    label: 'Clinic / Consultation',
    categories: ['Outpatient', 'Specialist Clinic'],
  },
  {
    id: 'emergency',
    label: 'Emergency',
    categories: ['Emergency'],
  },
  {
    id: 'maternity-delivery',
    label: 'Maternity / Delivery',
    categories: ['Delivery / Maternity'],
  },
  {
    id: 'tests-imaging',
    label: 'Tests & Imaging',
    categories: ['Radiology / Imaging', 'Laboratory', 'Procedures'],
  },
  {
    id: 'rehabilitation-therapy',
    label: 'Rehabilitation / Therapy',
    categories: [
      'Physiotherapy',
      'Occupational Therapy',
      'Speech Therapy',
      'Audiology',
      'Dietetics',
      'Traditional and Complementary Medicine',
    ],
  },
  {
    id: 'other-services',
    label: 'Other Services',
    categories: [
      'Medical Reports',
      'Nephrology',
      'Radiotherapy / Oncology',
      'Treatment Charges',
      'Other Published Charges',
    ],
  },
];

const normalizeCategory = (category: string): string =>
  category.trim().toLocaleLowerCase();

export const getAvailableHealthcareServiceGroups = (
  availableCategories: readonly string[],
): AvailableHealthcareServiceGroup[] => {
  const categoriesByNormalizedName = new Map(
    availableCategories
      .map((category) => category.trim())
      .filter(Boolean)
      .map((category) => [normalizeCategory(category), category]),
  );

  return HEALTHCARE_SERVICE_GROUPS.map((group) => ({
    ...group,
    categories: group.categories
      .map((category) =>
        categoriesByNormalizedName.get(normalizeCategory(category)),
      )
      .filter((category): category is string => Boolean(category)),
  })).filter((group) => group.categories.length > 0);
};
