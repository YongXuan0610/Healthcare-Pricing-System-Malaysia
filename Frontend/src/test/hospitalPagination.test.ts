import { describe, expect, it } from 'vitest';

import {
  hospitalPageSizeForViewport,
  paginateHospitals,
} from '@/lib/hospitalPagination';


const hospitals = ['HCJ', 'HSB', 'HPSF', 'HKL', 'HRC'];

describe('paginateHospitals', () => {
  it('shows two hospitals per desktop page and clamps the final page', () => {
    const first = paginateHospitals(hospitals, 1, 2);
    const second = paginateHospitals(hospitals, 2, 2);
    const third = paginateHospitals(hospitals, 3, 2);

    expect(first.visibleItems).toEqual(['HCJ', 'HSB']);
    expect(second.visibleItems).toEqual(['HPSF', 'HKL']);
    expect(third.visibleItems).toEqual(['HRC']);
    expect(first.pageCount).toBe(3);
    expect(first.hasPreviousPage).toBe(false);
    expect(first.hasNextPage).toBe(true);
    expect(third.hasPreviousPage).toBe(true);
    expect(third.hasNextPage).toBe(false);
    expect(third.showingStart).toBe(5);
    expect(third.showingEnd).toBe(5);
  });

  it('shows one hospital per mobile page', () => {
    const mobilePageSize = hospitalPageSizeForViewport(true);
    const page = paginateHospitals(hospitals, 3, mobilePageSize);

    expect(mobilePageSize).toBe(1);
    expect(page.pageCount).toBe(5);
    expect(page.visibleItems).toEqual(['HPSF']);
    expect(page.showingStart).toBe(3);
    expect(page.showingEnd).toBe(3);
  });

  it('uses two cards at the desktop/tablet breakpoint', () => {
    expect(hospitalPageSizeForViewport(false)).toBe(2);
  });

  it('clamps a page that becomes invalid after a responsive size change', () => {
    const resized = paginateHospitals(hospitals, 5, 2);

    expect(resized.currentPage).toBe(3);
    expect(resized.visibleItems).toEqual(['HRC']);
  });

  it('returns no page controls for empty or single-hospital results', () => {
    const empty = paginateHospitals<string>([], 1, 2);
    const single = paginateHospitals(['HCJ'], 1, 2);

    expect(empty.pageCount).toBe(0);
    expect(empty.visibleItems).toEqual([]);
    expect(single.pageCount).toBe(1);
    expect(single.hasPreviousPage).toBe(false);
    expect(single.hasNextPage).toBe(false);
  });
});
