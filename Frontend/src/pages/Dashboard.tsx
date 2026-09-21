import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Activity, History, User, Calculator, LogOut } from 'lucide-react';
import StatCard from '@/components/StatCard';
import { formatDateDisplay } from '@/lib/utils';
import { supabase } from '@/lib/supabase';

type Profile = {
  full_name: string;
};

type PredictionHistoryRow = {
  id: string;
  hospital_type: 'public' | 'private';
  total_cost: number | string;
  inputs: Record<string, unknown> | null;
  created_at: string;
};

const Dashboard = () => {
  const navigate = useNavigate();

  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [predictions, setPredictions] = useState<PredictionHistoryRow[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const loadDashboard = async () => {
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

        if (!isMounted) return;

        setEmail(user.email ?? '');

        const fallbackName =
          typeof user.user_metadata?.full_name === 'string'
            ? user.user_metadata.full_name
            : '';

        const [profileResult, predictionResult] = await Promise.all([
          supabase
            .from('profiles')
            .select('full_name')
            .eq('id', user.id)
            .maybeSingle<Profile>(),
          supabase
            .from('prediction_history')
            .select('id, hospital_type, total_cost, inputs, created_at')
            .eq('user_id', user.id)
            .order('created_at', { ascending: false }),
        ]);

        if (!isMounted) return;

        if (profileResult.error) {
          console.error('Profile loading error:', profileResult.error);
          setFullName(fallbackName);
        } else {
          setFullName(profileResult.data?.full_name || fallbackName);
        }

        if (predictionResult.error) {
          throw predictionResult.error;
        }

        setPredictions(
          (predictionResult.data ?? []) as PredictionHistoryRow[],
        );
      } catch (error) {
        console.error('Dashboard loading error:', error);

        if (isMounted) {
          setErrorMessage(
            'Unable to load your dashboard data. Please try again.',
          );
        }
      } finally {
        if (isMounted && !isRedirecting) {
          setIsLoading(false);
        }
      }
    };

    void loadDashboard();

    return () => {
      isMounted = false;
    };
  }, [navigate]);

  const publicPredictionCount = useMemo(
    () => predictions.filter((prediction) => prediction.hospital_type === 'public').length,
    [predictions],
  );

  const privatePredictionCount = useMemo(
    () => predictions.filter((prediction) => prediction.hospital_type === 'private').length,
    [predictions],
  );

  const recentPredictions = predictions.slice(0, 3);

  const firstName = fullName.trim()
    ? fullName.trim().split(/\s+/)[0]
    : 'User';

  const getPredictionLabel = (prediction: PredictionHistoryRow) => {
    const inputs = prediction.inputs ?? {};

    const category =
      typeof inputs.category === 'string' ? inputs.category : '';

    const packageName =
      typeof inputs.package === 'string' ? inputs.package : '';

    const treatment =
      typeof inputs.treatment === 'string' ? inputs.treatment : '';

    return category || packageName || treatment || 'Healthcare Cost Estimation';
  };

  const handleLogout = async () => {
    setIsSigningOut(true);
    setErrorMessage('');

    try {
      const { error } = await supabase.auth.signOut();

      if (error) {
        throw error;
      }

      navigate('/login', { replace: true });
    } catch (error) {
      console.error('Logout error:', error);
      setErrorMessage('Unable to sign out. Please try again.');
    } finally {
      setIsSigningOut(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-muted/30 flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-muted/30">
      {/* Top bar */}
      <header className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur-md">
        <div className="container flex h-16 items-center justify-between">
          <Link to="/" className="flex items-center gap-2 font-bold text-lg">
            <Activity className="h-5 w-5 text-primary" />
            MyCare<span className="text-primary">Cost</span>
          </Link>

          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground hidden sm:inline">
              {email}
            </span>

            <Button
              variant="ghost"
              size="sm"
              type="button"
              onClick={handleLogout}
              disabled={isSigningOut}
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <div className="container py-8">
        <h1 className="text-2xl font-bold mb-1">Welcome, {firstName}</h1>
        <p className="text-muted-foreground mb-8">
          Here's your prediction overview.
        </p>

        {errorMessage && (
          <div
            className="mb-6 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
            role="alert"
          >
            {errorMessage}
          </div>
        )}

        {/* Stats */}
        <div className="grid gap-4 sm:grid-cols-3 mb-8">
          <StatCard
            title="Total Predictions"
            value={predictions.length}
            icon={Calculator}
            description="all-time"
          />
          <StatCard
            title="Public Hospital"
            value={publicPredictionCount}
            icon={Activity}
          />
          <StatCard
            title="Private Hospital"
            value={privatePredictionCount}
            icon={Activity}
          />
        </div>

        {/* Quick actions */}
        <div className="flex flex-wrap gap-3 mb-8">
          <Button asChild>
            <Link to="/predict">
              <Calculator className="mr-2 h-4 w-4" />
              New Prediction
            </Link>
          </Button>

          <Button variant="outline" asChild>
            <Link to="/history">
              <History className="mr-2 h-4 w-4" />
              View History
            </Link>
          </Button>

          <Button variant="outline" asChild>
            <Link to="/profile">
              <User className="mr-2 h-4 w-4" />
              Profile
            </Link>
          </Button>
        </div>

        {/* Recent predictions */}
        <h2 className="text-lg font-semibold mb-4">Recent Predictions</h2>

        {recentPredictions.length === 0 ? (
          <div className="rounded-lg border bg-card p-6 card-shadow text-center">
            <p className="font-medium">No predictions yet</p>
            <p className="text-sm text-muted-foreground mt-1">
              Create your first healthcare cost prediction to see it here.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {recentPredictions.map((prediction) => (
              <div
                key={prediction.id}
                className="rounded-lg border bg-card p-4 card-shadow flex items-center justify-between gap-4"
              >
                <div>
                  <p className="font-medium text-sm capitalize">
                    {prediction.hospital_type} Hospital —{' '}
                    {getPredictionLabel(prediction)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {formatDateDisplay(prediction.created_at)}
                  </p>
                </div>

                <p className="font-bold text-primary whitespace-nowrap">
                  RM {Number(prediction.total_cost).toLocaleString('en-MY')}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
