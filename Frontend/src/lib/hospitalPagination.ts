export interface HospitalPage<T> {
  currentPage: number;
  pageCount: number;
  visibleItems: T[];
  showingStart: number;
  showingEnd: number;
  hasPreviousPage: boolean;
  hasNextPage: boolean;
}

export const hospitalPageSizeForViewport = (isMobile: boolean): 1 | 2 =>
  isMobile ? 1 : 2;

export const paginateHospitals = <T>(
  hospitals: T[],
  requestedPage: number,
  requestedPageSize: number,
): HospitalPage<T> => {
  const pageSize = Math.max(1, Math.floor(requestedPageSize) || 1);
  const pageCount = Math.ceil(hospitals.length / pageSize);
  const normalizedPage = Math.max(1, Math.floor(requestedPage) || 1);
  const currentPage = pageCount === 0
    ? 1
    : Math.min(normalizedPage, pageCount);
  const startIndex = (currentPage - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, hospitals.length);

  return {
    currentPage,
    pageCount,
    visibleItems: hospitals.slice(startIndex, endIndex),
    showingStart: hospitals.length === 0 ? 0 : startIndex + 1,
    showingEnd: endIndex,
    hasPreviousPage: pageCount > 0 && currentPage > 1,
    hasNextPage: currentPage < pageCount,
  };
};
