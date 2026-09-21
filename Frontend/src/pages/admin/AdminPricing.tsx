import { useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Pencil, Trash2, Plus, Search } from 'lucide-react';
import { supabase } from '@/lib/supabase';

type PrivatePackage = {
  id: number;
  hospital: string;
  package_name: string;
  price_rm: number | string;
  gender_target: string | null;
  age_target: string | null;
};

type WardRate = {
  id: number;
  hospital_key: string;
  hospital: string;
  branch_location: string | null;
  ward_type_original: string;
  normalised_category: string | null;
  price_rm: number | string;
  rate_basis: string | null;
  official_source_url: string | null;
  accessed_date: string | null;
};

type PublicCharge = {
  id: number;
  category: string;
  service_name: string;
  patient_class: string;
  charge_type: string;
  price_rm: number | string;
  source_url: string | null;
};

type PackageForm = {
  hospital: string;
  package_name: string;
  price_rm: string;
  gender_target: string;
  age_target: string;
};

type WardForm = {
  hospital_key: string;
  hospital: string;
  branch_location: string;
  ward_type_original: string;
  normalised_category: string;
  price_rm: string;
  rate_basis: string;
  official_source_url: string;
  accessed_date: string;
};

type PublicChargeForm = {
  category: string;
  service_name: string;
  patient_class: string;
  charge_type: string;
  price_rm: string;
  source_url: string;
};

const emptyPackageForm: PackageForm = {
  hospital: '',
  package_name: '',
  price_rm: '',
  gender_target: 'Both',
  age_target: 'Adult',
};

const emptyWardForm: WardForm = {
  hospital_key: '',
  hospital: '',
  branch_location: '',
  ward_type_original: '',
  normalised_category: '',
  price_rm: '',
  rate_basis: 'Per day',
  official_source_url: '',
  accessed_date: '',
};

const emptyPublicChargeForm: PublicChargeForm = {
  category: '',
  service_name: '',
  patient_class: 'all',
  charge_type: '',
  price_rm: '',
  source_url: '',
};

const AdminPricing = () => {
  const [tab, setTab] = useState('packages');
  const [search, setSearch] = useState('');

  const [packages, setPackages] = useState<PrivatePackage[]>([]);
  const [wards, setWards] = useState<WardRate[]>([]);
  const [publicCharges, setPublicCharges] = useState<PublicCharge[]>([]);

  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [deletingKey, setDeletingKey] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const [packageDialogOpen, setPackageDialogOpen] = useState(false);
  const [editingPackageId, setEditingPackageId] = useState<number | null>(null);
  const [packageForm, setPackageForm] = useState<PackageForm>(emptyPackageForm);

  const [wardDialogOpen, setWardDialogOpen] = useState(false);
  const [editingWardId, setEditingWardId] = useState<number | null>(null);
  const [wardForm, setWardForm] = useState<WardForm>(emptyWardForm);

  const [publicDialogOpen, setPublicDialogOpen] = useState(false);
  const [editingPublicId, setEditingPublicId] = useState<number | null>(null);
  const [publicForm, setPublicForm] = useState<PublicChargeForm>(emptyPublicChargeForm);

  const loadPricingData = async () => {
    setIsLoading(true);
    setErrorMessage('');

    try {
      const [packageResult, wardResult, publicResult] = await Promise.all([
        supabase
          .from('private_packages')
          .select('id, hospital, package_name, price_rm, gender_target, age_target')
          .order('hospital')
          .order('package_name'),
        supabase
          .from('private_ward_rates')
          .select(
            'id, hospital_key, hospital, branch_location, ward_type_original, normalised_category, price_rm, rate_basis, official_source_url, accessed_date',
          )
          .order('hospital_key')
          .order('price_rm'),
        supabase
          .from('public_charges')
          .select('id, category, service_name, patient_class, charge_type, price_rm, source_url')
          .order('category')
          .order('service_name'),
      ]);

      if (packageResult.error) throw packageResult.error;
      if (wardResult.error) throw wardResult.error;
      if (publicResult.error) throw publicResult.error;

      setPackages((packageResult.data ?? []) as PrivatePackage[]);
      setWards((wardResult.data ?? []) as WardRate[]);
      setPublicCharges((publicResult.data ?? []) as PublicCharge[]);
    } catch (error) {
      console.error('Pricing data loading error:', error);
      setErrorMessage('Unable to load pricing data from Supabase.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadPricingData();
  }, []);

  useEffect(() => {
    setSearch('');
    setErrorMessage('');
    setSuccessMessage('');
  }, [tab]);

  const filteredPackages = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return packages;

    return packages.filter((item) =>
      [item.hospital, item.package_name, item.gender_target ?? '', item.age_target ?? '']
        .some((value) => value.toLowerCase().includes(query)),
    );
  }, [packages, search]);

  const filteredWards = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return wards;

    return wards.filter((item) =>
      [
        item.hospital_key,
        item.hospital,
        item.branch_location ?? '',
        item.ward_type_original,
        item.normalised_category ?? '',
      ].some((value) => value.toLowerCase().includes(query)),
    );
  }, [wards, search]);

  const filteredPublicCharges = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return publicCharges;

    return publicCharges.filter((item) =>
      [item.category, item.service_name, item.patient_class, item.charge_type]
        .some((value) => value.toLowerCase().includes(query)),
    );
  }, [publicCharges, search]);

  const resetMessages = () => {
    setErrorMessage('');
    setSuccessMessage('');
  };

  const openAddPackage = () => {
    resetMessages();
    setEditingPackageId(null);
    setPackageForm(emptyPackageForm);
    setPackageDialogOpen(true);
  };

  const openEditPackage = (item: PrivatePackage) => {
    resetMessages();
    setEditingPackageId(item.id);
    setPackageForm({
      hospital: item.hospital,
      package_name: item.package_name,
      price_rm: String(item.price_rm),
      gender_target: item.gender_target ?? '',
      age_target: item.age_target ?? '',
    });
    setPackageDialogOpen(true);
  };

  const savePackage = async (event: React.FormEvent) => {
    event.preventDefault();

    const price = Number(packageForm.price_rm);
    if (!packageForm.hospital.trim() || !packageForm.package_name.trim()) {
      setErrorMessage('Hospital and package name are required.');
      return;
    }
    if (!Number.isFinite(price) || price < 0) {
      setErrorMessage('Package price must be a valid non-negative number.');
      return;
    }

    setIsSaving(true);
    resetMessages();

    try {
      const payload = {
        hospital: packageForm.hospital.trim(),
        package_name: packageForm.package_name.trim(),
        price_rm: price,
        gender_target: packageForm.gender_target.trim() || null,
        age_target: packageForm.age_target.trim() || null,
        updated_at: new Date().toISOString(),
      };

      const result = editingPackageId === null
        ? await supabase.from('private_packages').insert(payload)
        : await supabase.from('private_packages').update(payload).eq('id', editingPackageId);

      if (result.error) throw result.error;

      setPackageDialogOpen(false);
      setSuccessMessage(
        editingPackageId === null
          ? 'Private package added successfully.'
          : 'Private package updated successfully.',
      );
      await loadPricingData();
    } catch (error) {
      console.error('Package save error:', error);
      setErrorMessage('Unable to save the private package.');
    } finally {
      setIsSaving(false);
    }
  };

  const openAddWard = () => {
    resetMessages();
    setEditingWardId(null);
    setWardForm(emptyWardForm);
    setWardDialogOpen(true);
  };

  const openEditWard = (item: WardRate) => {
    resetMessages();
    setEditingWardId(item.id);
    setWardForm({
      hospital_key: item.hospital_key,
      hospital: item.hospital,
      branch_location: item.branch_location ?? '',
      ward_type_original: item.ward_type_original,
      normalised_category: item.normalised_category ?? '',
      price_rm: String(item.price_rm),
      rate_basis: item.rate_basis ?? '',
      official_source_url: item.official_source_url ?? '',
      accessed_date: item.accessed_date ?? '',
    });
    setWardDialogOpen(true);
  };

  const saveWard = async (event: React.FormEvent) => {
    event.preventDefault();

    const price = Number(wardForm.price_rm);
    if (
      !wardForm.hospital_key.trim() ||
      !wardForm.hospital.trim() ||
      !wardForm.ward_type_original.trim()
    ) {
      setErrorMessage('Hospital key, hospital name, and ward type are required.');
      return;
    }
    if (!Number.isFinite(price) || price < 0) {
      setErrorMessage('Ward price must be a valid non-negative number.');
      return;
    }

    setIsSaving(true);
    resetMessages();

    try {
      const payload = {
        hospital_key: wardForm.hospital_key.trim(),
        hospital: wardForm.hospital.trim(),
        branch_location: wardForm.branch_location.trim() || null,
        ward_type_original: wardForm.ward_type_original.trim(),
        normalised_category: wardForm.normalised_category.trim() || null,
        price_rm: price,
        rate_basis: wardForm.rate_basis.trim() || null,
        official_source_url: wardForm.official_source_url.trim() || null,
        accessed_date: wardForm.accessed_date || null,
        updated_at: new Date().toISOString(),
      };

      const result = editingWardId === null
        ? await supabase.from('private_ward_rates').insert(payload)
        : await supabase.from('private_ward_rates').update(payload).eq('id', editingWardId);

      if (result.error) throw result.error;

      setWardDialogOpen(false);
      setSuccessMessage(
        editingWardId === null
          ? 'Ward rate added successfully.'
          : 'Ward rate updated successfully.',
      );
      await loadPricingData();
    } catch (error) {
      console.error('Ward save error:', error);
      setErrorMessage('Unable to save the ward rate.');
    } finally {
      setIsSaving(false);
    }
  };

  const openAddPublicCharge = () => {
    resetMessages();
    setEditingPublicId(null);
    setPublicForm(emptyPublicChargeForm);
    setPublicDialogOpen(true);
  };

  const openEditPublicCharge = (item: PublicCharge) => {
    resetMessages();
    setEditingPublicId(item.id);
    setPublicForm({
      category: item.category,
      service_name: item.service_name,
      patient_class: item.patient_class,
      charge_type: item.charge_type,
      price_rm: String(item.price_rm),
      source_url: item.source_url ?? '',
    });
    setPublicDialogOpen(true);
  };

  const savePublicCharge = async (event: React.FormEvent) => {
    event.preventDefault();

    const price = Number(publicForm.price_rm);
    if (
      !publicForm.category.trim() ||
      !publicForm.service_name.trim() ||
      !publicForm.patient_class.trim() ||
      !publicForm.charge_type.trim()
    ) {
      setErrorMessage('Category, service, patient class, and charge type are required.');
      return;
    }
    if (!Number.isFinite(price) || price < 0) {
      setErrorMessage('Public charge must be a valid non-negative number.');
      return;
    }

    setIsSaving(true);
    resetMessages();

    try {
      const payload = {
        category: publicForm.category.trim().toLowerCase(),
        service_name: publicForm.service_name.trim(),
        patient_class: publicForm.patient_class.trim().toLowerCase(),
        charge_type: publicForm.charge_type.trim(),
        price_rm: price,
        source_url: publicForm.source_url.trim() || null,
        updated_at: new Date().toISOString(),
      };

      const result = editingPublicId === null
        ? await supabase.from('public_charges').insert(payload)
        : await supabase.from('public_charges').update(payload).eq('id', editingPublicId);

      if (result.error) throw result.error;

      setPublicDialogOpen(false);
      setSuccessMessage(
        editingPublicId === null
          ? 'Public charge added successfully.'
          : 'Public charge updated successfully.',
      );
      await loadPricingData();
    } catch (error) {
      console.error('Public charge save error:', error);
      setErrorMessage('Unable to save the public charge.');
    } finally {
      setIsSaving(false);
    }
  };

  const deleteRow = async (
    table: 'private_packages' | 'private_ward_rates' | 'public_charges',
    id: number,
    label: string,
  ) => {
    const confirmed = window.confirm(`Delete ${label}?`);
    if (!confirmed) return;

    const key = `${table}-${id}`;
    setDeletingKey(key);
    resetMessages();

    try {
      const { error } = await supabase.from(table).delete().eq('id', id);
      if (error) throw error;

      setSuccessMessage('Pricing record deleted successfully.');
      await loadPricingData();
    } catch (error) {
      console.error('Pricing delete error:', error);
      setErrorMessage('Unable to delete the pricing record.');
    } finally {
      setDeletingKey(null);
    }
  };

  return (
    <div className="p-6 md:p-8">
      <h1 className="text-2xl font-bold mb-1">Pricing &amp; Content Management</h1>
      <p className="text-muted-foreground mb-6">
        Manage the PostgreSQL pricing records used by MyCareCost.
      </p>

      {errorMessage && (
        <div
          className="mb-4 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
          role="alert"
        >
          {errorMessage}
        </div>
      )}

      {successMessage && (
        <div
          className="mb-4 rounded-lg border border-healthcare-green/25 bg-healthcare-green-light p-3 text-sm text-healthcare-green"
          role="status"
        >
          {successMessage}
        </div>
      )}

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="mb-6">
          <TabsTrigger value="packages">Packages ({packages.length})</TabsTrigger>
          <TabsTrigger value="wards">Ward Fees ({wards.length})</TabsTrigger>
          <TabsTrigger value="public">Public Charges ({publicCharges.length})</TabsTrigger>
        </TabsList>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between mb-4">
          <div className="relative w-full sm:max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search current pricing records..."
              className="pl-9"
            />
          </div>

          {tab === 'packages' && (
            <Button size="sm" onClick={openAddPackage}>
              <Plus className="mr-2 h-3.5 w-3.5" /> Add Package
            </Button>
          )}
          {tab === 'wards' && (
            <Button size="sm" onClick={openAddWard}>
              <Plus className="mr-2 h-3.5 w-3.5" /> Add Ward
            </Button>
          )}
          {tab === 'public' && (
            <Button size="sm" onClick={openAddPublicCharge}>
              <Plus className="mr-2 h-3.5 w-3.5" /> Add Public Charge
            </Button>
          )}
        </div>

        <TabsContent value="packages">
          <div className="rounded-xl border bg-card card-shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Hospital</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Package</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden md:table-cell">Target</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Price (RM)</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    <tr><td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">Loading packages...</td></tr>
                  ) : filteredPackages.length === 0 ? (
                    <tr><td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">No packages found.</td></tr>
                  ) : filteredPackages.map((item) => (
                    <tr key={item.id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="px-4 py-3 font-medium">{item.hospital}</td>
                      <td className="px-4 py-3">{item.package_name}</td>
                      <td className="px-4 py-3 text-muted-foreground hidden md:table-cell">
                        {(item.gender_target || 'Any')} / {(item.age_target || 'Any')}
                      </td>
                      <td className="px-4 py-3 text-right font-medium">{Number(item.price_rm).toLocaleString('en-MY', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => openEditPackage(item)} title="Edit package">
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => deleteRow('private_packages', item.id, item.package_name)} disabled={deletingKey === `private_packages-${item.id}`} title="Delete package">
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="wards">
          <div className="rounded-xl border bg-card card-shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Hospital</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Ward / Room</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden md:table-cell">Category</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Price (RM)</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    <tr><td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">Loading ward rates...</td></tr>
                  ) : filteredWards.length === 0 ? (
                    <tr><td colSpan={5} className="px-4 py-10 text-center text-muted-foreground">No ward rates found.</td></tr>
                  ) : filteredWards.map((item) => (
                    <tr key={item.id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="px-4 py-3">
                        <p className="font-medium">{item.hospital_key}</p>
                        <p className="text-xs text-muted-foreground">{item.hospital}</p>
                      </td>
                      <td className="px-4 py-3">{item.ward_type_original}</td>
                      <td className="px-4 py-3 text-muted-foreground hidden md:table-cell">{item.normalised_category || '—'}</td>
                      <td className="px-4 py-3 text-right font-medium">{Number(item.price_rm).toLocaleString('en-MY', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => openEditWard(item)} title="Edit ward rate">
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => deleteRow('private_ward_rates', item.id, item.ward_type_original)} disabled={deletingKey === `private_ward_rates-${item.id}`} title="Delete ward rate">
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="public">
          <div className="rounded-xl border bg-card card-shadow overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Category</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Service</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden md:table-cell">Patient Class</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">Charge Type</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Price (RM)</th>
                    <th className="text-right px-4 py-3 font-medium text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {isLoading ? (
                    <tr><td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">Loading public charges...</td></tr>
                  ) : filteredPublicCharges.length === 0 ? (
                    <tr><td colSpan={6} className="px-4 py-10 text-center text-muted-foreground">No public charges found.</td></tr>
                  ) : filteredPublicCharges.map((item) => (
                    <tr key={item.id} className="border-b last:border-0 hover:bg-muted/30">
                      <td className="px-4 py-3 capitalize">{item.category}</td>
                      <td className="px-4 py-3 font-medium">{item.service_name}</td>
                      <td className="px-4 py-3 text-muted-foreground hidden md:table-cell capitalize">{item.patient_class}</td>
                      <td className="px-4 py-3 text-muted-foreground hidden lg:table-cell">{item.charge_type}</td>
                      <td className="px-4 py-3 text-right font-medium">{Number(item.price_rm).toLocaleString('en-MY', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                      <td className="px-4 py-3 text-right whitespace-nowrap">
                        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => openEditPublicCharge(item)} title="Edit public charge">
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive" onClick={() => deleteRow('public_charges', item.id, item.service_name)} disabled={deletingKey === `public_charges-${item.id}`} title="Delete public charge">
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </TabsContent>
      </Tabs>

      <Dialog open={packageDialogOpen} onOpenChange={setPackageDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{editingPackageId === null ? 'Add Private Package' : 'Edit Private Package'}</DialogTitle>
            <DialogDescription>Maintain a published private-hospital package record.</DialogDescription>
          </DialogHeader>
          <form onSubmit={savePackage} className="space-y-4">
            <div className="space-y-2">
              <Label>Hospital</Label>
              <Input value={packageForm.hospital} onChange={(e) => setPackageForm((current) => ({ ...current, hospital: e.target.value }))} required />
            </div>
            <div className="space-y-2">
              <Label>Package Name</Label>
              <Input value={packageForm.package_name} onChange={(e) => setPackageForm((current) => ({ ...current, package_name: e.target.value }))} required />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Gender Target</Label>
                <Input value={packageForm.gender_target} onChange={(e) => setPackageForm((current) => ({ ...current, gender_target: e.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label>Age Target</Label>
                <Input value={packageForm.age_target} onChange={(e) => setPackageForm((current) => ({ ...current, age_target: e.target.value }))} />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Price (RM)</Label>
              <Input type="number" min="0" step="0.01" value={packageForm.price_rm} onChange={(e) => setPackageForm((current) => ({ ...current, price_rm: e.target.value }))} required />
            </div>
            <DialogFooter>
              <Button type="submit" disabled={isSaving}>{isSaving ? 'Saving...' : 'Save Package'}</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={wardDialogOpen} onOpenChange={setWardDialogOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{editingWardId === null ? 'Add Ward Rate' : 'Edit Ward Rate'}</DialogTitle>
            <DialogDescription>Maintain a private-hospital ward or room rate.</DialogDescription>
          </DialogHeader>
          <form onSubmit={saveWard} className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Hospital Key</Label>
                <Input placeholder="e.g. Sunway" value={wardForm.hospital_key} onChange={(e) => setWardForm((current) => ({ ...current, hospital_key: e.target.value }))} required />
              </div>
              <div className="space-y-2">
                <Label>Branch / Location</Label>
                <Input value={wardForm.branch_location} onChange={(e) => setWardForm((current) => ({ ...current, branch_location: e.target.value }))} />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Official Hospital Name</Label>
              <Input value={wardForm.hospital} onChange={(e) => setWardForm((current) => ({ ...current, hospital: e.target.value }))} required />
            </div>
            <div className="space-y-2">
              <Label>Ward / Room Type</Label>
              <Input value={wardForm.ward_type_original} onChange={(e) => setWardForm((current) => ({ ...current, ward_type_original: e.target.value }))} required />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Normalised Category</Label>
                <Input value={wardForm.normalised_category} onChange={(e) => setWardForm((current) => ({ ...current, normalised_category: e.target.value }))} />
              </div>
              <div className="space-y-2">
                <Label>Rate Basis</Label>
                <Input value={wardForm.rate_basis} onChange={(e) => setWardForm((current) => ({ ...current, rate_basis: e.target.value }))} />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Price (RM)</Label>
              <Input type="number" min="0" step="0.01" value={wardForm.price_rm} onChange={(e) => setWardForm((current) => ({ ...current, price_rm: e.target.value }))} required />
            </div>
            <div className="space-y-2">
              <Label>Official Source URL</Label>
              <Input type="url" value={wardForm.official_source_url} onChange={(e) => setWardForm((current) => ({ ...current, official_source_url: e.target.value }))} />
            </div>
            <div className="space-y-2">
              <Label>Accessed Date</Label>
              <Input type="date" value={wardForm.accessed_date} onChange={(e) => setWardForm((current) => ({ ...current, accessed_date: e.target.value }))} />
            </div>
            <DialogFooter>
              <Button type="submit" disabled={isSaving}>{isSaving ? 'Saving...' : 'Save Ward Rate'}</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog open={publicDialogOpen} onOpenChange={setPublicDialogOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{editingPublicId === null ? 'Add Public Charge' : 'Edit Public Charge'}</DialogTitle>
            <DialogDescription>Maintain a published public-hospital charge record.</DialogDescription>
          </DialogHeader>
          <form onSubmit={savePublicCharge} className="space-y-4">
            <div className="space-y-2">
              <Label>Category</Label>
              <Input value={publicForm.category} onChange={(e) => setPublicForm((current) => ({ ...current, category: e.target.value }))} required />
            </div>
            <div className="space-y-2">
              <Label>Service Name</Label>
              <Input value={publicForm.service_name} onChange={(e) => setPublicForm((current) => ({ ...current, service_name: e.target.value }))} required />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Patient Class</Label>
                <Input value={publicForm.patient_class} onChange={(e) => setPublicForm((current) => ({ ...current, patient_class: e.target.value }))} required />
              </div>
              <div className="space-y-2">
                <Label>Charge Type</Label>
                <Input value={publicForm.charge_type} onChange={(e) => setPublicForm((current) => ({ ...current, charge_type: e.target.value }))} required />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Price (RM)</Label>
              <Input type="number" min="0" step="0.01" value={publicForm.price_rm} onChange={(e) => setPublicForm((current) => ({ ...current, price_rm: e.target.value }))} required />
            </div>
            <div className="space-y-2">
              <Label>Source URL</Label>
              <Input type="url" value={publicForm.source_url} onChange={(e) => setPublicForm((current) => ({ ...current, source_url: e.target.value }))} />
            </div>
            <DialogFooter>
              <Button type="submit" disabled={isSaving}>{isSaving ? 'Saving...' : 'Save Public Charge'}</Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default AdminPricing;
