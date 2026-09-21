import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { ArrowLeft, Save } from 'lucide-react';
import PageHeader from '@/components/PageHeader';
import { MALAYSIAN_STATES } from '@/lib/referenceData';
import { supabase } from '@/lib/supabase';

const Profile = () => {
  const navigate = useNavigate();

  const [userId, setUserId] = useState('');
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [ic, setIc] = useState('');
  const [phone, setPhone] = useState('');
  const [state, setState] = useState('');

  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const loadProfile = async () => {
      let isRedirecting = false;
      setIsLoading(true);
      setErrorMessage('');

      try {
        const {
          data: { user },
          error: userError,
        } = await supabase.auth.getUser();

        if (!user) {
          isRedirecting = true;
          navigate('/login', { replace: true });
          return;
        }

        if (userError) {
          throw userError;
        }

        const { data: profile, error: profileError } = await supabase
          .from('profiles')
          .select('full_name, ic_number, phone, state')
          .eq('id', user.id)
          .maybeSingle();

        if (profileError) {
          throw profileError;
        }

        if (!isMounted) return;

        setUserId(user.id);
        setEmail(user.email ?? '');
        setName(
          profile?.full_name ||
            (typeof user.user_metadata?.full_name === 'string'
              ? user.user_metadata.full_name
              : ''),
        );
        setIc(
          profile?.ic_number ||
            (typeof user.user_metadata?.ic_number === 'string'
              ? user.user_metadata.ic_number
              : ''),
        );
        setPhone(
          profile?.phone ||
            (typeof user.user_metadata?.phone === 'string'
              ? user.user_metadata.phone
              : ''),
        );
        setState(
          profile?.state ||
            (typeof user.user_metadata?.state === 'string'
              ? user.user_metadata.state
              : ''),
        );
      } catch (error) {
        console.error('Profile loading error:', error);

        if (isMounted) {
          setErrorMessage('Unable to load your profile. Please try again.');
        }
      } finally {
        if (isMounted && !isRedirecting) {
          setIsLoading(false);
        }
      }
    };

    void loadProfile();

    return () => {
      isMounted = false;
    };
  }, [navigate]);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!userId) {
      setErrorMessage('No authenticated user was found. Please sign in again.');
      return;
    }

    if (!name.trim()) {
      setErrorMessage('Full name is required.');
      return;
    }

    if (!state) {
      setErrorMessage('Please select your state.');
      return;
    }

    setIsSaving(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const { error } = await supabase
        .from('profiles')
        .upsert(
          {
            id: userId,
            full_name: name.trim(),
            ic_number: ic.trim() || null,
            phone: phone.trim() || null,
            state,
            updated_at: new Date().toISOString(),
          },
          { onConflict: 'id' },
        );

      if (error) {
        throw error;
      }

      setSuccessMessage('Profile updated successfully.');
    } catch (error) {
      console.error('Profile save error:', error);
      setErrorMessage('Unable to save your profile. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-muted/30 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading profile...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-muted/30">
      <header className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur-md">
        <div className="container flex h-16 items-center">
          <Link
            to="/dashboard"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Link>
        </div>
      </header>

      <PageHeader
        title="My Profile"
        description="Manage your personal information."
      />

      <div className="container py-8 max-w-lg">
        <form
          onSubmit={handleSave}
          className="rounded-xl border bg-card p-6 md:p-8 card-shadow space-y-4"
        >
          <div className="space-y-2">
            <Label htmlFor="fullName">Full Name</Label>
            <Input
              id="fullName"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              disabled={isSaving}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              value={email}
              disabled
              className="bg-muted"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="icNumber">IC Number</Label>
            <Input
              id="icNumber"
              value={ic}
              onChange={(e) => setIc(e.target.value)}
              placeholder="e.g. 901015-01-1234"
              disabled={isSaving}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="phoneNumber">Phone Number</Label>
            <Input
              id="phoneNumber"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="e.g. 012-3456789"
              disabled={isSaving}
            />
          </div>

          <div className="space-y-2">
            <Label>State</Label>
            <Select
              value={state}
              onValueChange={setState}
              disabled={isSaving}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select state" />
              </SelectTrigger>
              <SelectContent>
                {MALAYSIAN_STATES.map((malaysianState) => (
                  <SelectItem
                    key={malaysianState}
                    value={malaysianState}
                  >
                    {malaysianState}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {errorMessage && (
            <p className="text-sm text-destructive" role="alert">
              {errorMessage}
            </p>
          )}

          {successMessage && (
            <p className="text-sm text-healthcare-green" role="status">
              {successMessage}
            </p>
          )}

          <Button
            type="submit"
            className="w-full"
            disabled={isSaving}
          >
            <Save className="mr-2 h-4 w-4" />
            {isSaving ? 'Saving...' : 'Save Changes'}
          </Button>
        </form>
      </div>
    </div>
  );
};

export default Profile;
