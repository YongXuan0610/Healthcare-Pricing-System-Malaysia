import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Save, Settings2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';

type SystemSettings = {
  id: number;
  system_name: string;
  contact_email: string;
  allow_guest_predictions: boolean;
  maintenance_mode: boolean;
  updated_at: string;
};

const DEFAULT_SETTINGS: SystemSettings = {
  id: 1,
  system_name: 'MyCareCost',
  contact_email: 'admin@mycarecost.com',
  allow_guest_predictions: true,
  maintenance_mode: false,
  updated_at: '',
};

const AdminSettings = () => {
  const [settings, setSettings] = useState<SystemSettings>(DEFAULT_SETTINGS);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const loadSettings = async () => {
      setIsLoading(true);
      setErrorMessage('');

      try {
        const { data, error } = await supabase
          .from('system_settings')
          .select(
            'id,system_name,contact_email,allow_guest_predictions,maintenance_mode,updated_at',
          )
          .eq('id', 1)
          .single();

        if (error) {
          throw error;
        }

        if (isMounted && data) {
          setSettings(data as SystemSettings);
        }
      } catch (error) {
        console.error('System settings loading error:', error);

        if (isMounted) {
          setErrorMessage('Unable to load system settings.');
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    void loadSettings();

    return () => {
      isMounted = false;
    };
  }, []);

  const handleSave = async () => {
    setErrorMessage('');
    setSuccessMessage('');

    const systemName = settings.system_name.trim();
    const contactEmail = settings.contact_email.trim();

    if (!systemName) {
      setErrorMessage('System name is required.');
      return;
    }

    if (!contactEmail) {
      setErrorMessage('Contact email is required.');
      return;
    }

    setIsSaving(true);

    try {
      const {
        data: { user },
        error: userError,
      } = await supabase.auth.getUser();

      if (userError) {
        throw userError;
      }

      if (!user) {
        setErrorMessage('Your administrator session has expired.');
        return;
      }

      const updatedAt = new Date().toISOString();

      const { data, error } = await supabase
        .from('system_settings')
        .update({
          system_name: systemName,
          contact_email: contactEmail,
          allow_guest_predictions: settings.allow_guest_predictions,
          maintenance_mode: settings.maintenance_mode,
          updated_at: updatedAt,
          updated_by: user.id,
        })
        .eq('id', 1)
        .select(
          'id,system_name,contact_email,allow_guest_predictions,maintenance_mode,updated_at',
        )
        .single();

      if (error) {
        throw error;
      }

      setSettings(data as SystemSettings);
      setSuccessMessage('System settings saved successfully.');
    } catch (error) {
      console.error('System settings save error:', error);
      setErrorMessage(
        'Unable to save system settings. Confirm that you are logged in as an administrator.',
      );
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="p-6 md:p-8">
      <h1 className="text-2xl font-bold mb-1">System Settings</h1>
      <p className="text-muted-foreground mb-8">
        Configure system-wide options stored in Supabase.
      </p>

      <div className="max-w-xl space-y-6">
        {errorMessage && (
          <div
            className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
            role="alert"
          >
            {errorMessage}
          </div>
        )}

        {successMessage && (
          <div
            className="rounded-lg border border-healthcare-green/25 bg-healthcare-green-light p-3 text-sm text-healthcare-green"
            role="status"
          >
            {successMessage}
          </div>
        )}

        <div className="rounded-xl border bg-card p-6 card-shadow space-y-5">
          <div className="flex items-center gap-2">
            <Settings2 className="h-4 w-4 text-primary" />
            <h2 className="font-semibold">General</h2>
          </div>

          <div className="space-y-2">
            <Label htmlFor="system-name">System Name</Label>
            <Input
              id="system-name"
              value={settings.system_name}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  system_name: event.target.value,
                }))
              }
              disabled={isLoading || isSaving}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="contact-email">Contact Email</Label>
            <Input
              id="contact-email"
              type="email"
              value={settings.contact_email}
              onChange={(event) =>
                setSettings((current) => ({
                  ...current,
                  contact_email: event.target.value,
                }))
              }
              disabled={isLoading || isSaving}
            />
          </div>

          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium">Guest Predictions</p>
              <p className="text-xs text-muted-foreground">
                Allow visitors to use pricing tools without signing in.
              </p>
            </div>

            <Switch
              checked={settings.allow_guest_predictions}
              onCheckedChange={(checked) =>
                setSettings((current) => ({
                  ...current,
                  allow_guest_predictions: checked,
                }))
              }
              disabled={isLoading || isSaving}
            />
          </div>

          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-medium">Maintenance Mode</p>
              <p className="text-xs text-muted-foreground">
                Temporarily disable user-facing pricing features while keeping
                the admin portal available.
              </p>
            </div>

            <Switch
              checked={settings.maintenance_mode}
              onCheckedChange={(checked) =>
                setSettings((current) => ({
                  ...current,
                  maintenance_mode: checked,
                }))
              }
              disabled={isLoading || isSaving}
            />
          </div>
        </div>

        <div className="rounded-xl border bg-card p-4 text-sm text-muted-foreground">
          {settings.updated_at
            ? `Last saved: ${new Date(settings.updated_at).toLocaleString()}`
            : 'Settings have not been loaded yet.'}
        </div>

        <Button
          type="button"
          size="lg"
          onClick={handleSave}
          disabled={isLoading || isSaving}
        >
          <Save className="mr-2 h-4 w-4" />
          {isSaving ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>
    </div>
  );
};

export default AdminSettings;
