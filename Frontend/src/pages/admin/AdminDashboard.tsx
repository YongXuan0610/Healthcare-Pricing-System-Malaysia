import { useEffect, useState } from 'react';
import StatCard from '@/components/StatCard';
import { Users, Calculator, Building2, Building } from 'lucide-react';
import { formatDateDisplay } from '@/lib/utils';
import { supabase } from '@/lib/supabase';

type RecentUser = {
  id: string;
  full_name: string;
  email: string | null;
  state: string | null;
  created_at: string;
};

type RecentPrediction = {
  id: string;
  hospital_type: 'public' | 'private';
  total_cost: number | string;
  currency: string;
  created_at: string;
  full_name: string | null;
  email: string | null;
};

type AdminDashboardData = {
  total_users: number;
  total_predictions: number;
  public_percentage: number;
  private_percentage: number;
  recent_users: RecentUser[];
  recent_predictions: RecentPrediction[];
};

const emptyDashboard: AdminDashboardData = {
  total_users: 0,
  total_predictions: 0,
  public_percentage: 0,
  private_percentage: 0,
  recent_users: [],
  recent_predictions: [],
};

const AdminDashboard = () => {
  const [dashboard, setDashboard] =
    useState<AdminDashboardData>(emptyDashboard);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const loadDashboard = async () => {
      setIsLoading(true);
      setErrorMessage('');

      try {
        const { data, error } = await supabase.rpc(
          'get_admin_dashboard_data',
        );

        if (error) {
          throw error;
        }

        if (!isMounted) {
          return;
        }

        const response = (data ?? {}) as Partial<AdminDashboardData>;

        setDashboard({
          total_users: Number(response.total_users ?? 0),
          total_predictions: Number(response.total_predictions ?? 0),
          public_percentage: Number(response.public_percentage ?? 0),
          private_percentage: Number(response.private_percentage ?? 0),
          recent_users: Array.isArray(response.recent_users)
            ? response.recent_users
            : [],
          recent_predictions: Array.isArray(response.recent_predictions)
            ? response.recent_predictions
            : [],
        });
      } catch (error) {
        console.error('Admin dashboard loading error:', error);

        if (isMounted) {
          setErrorMessage(
            'Unable to load admin analytics. Please try again.',
          );
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    void loadDashboard();

    return () => {
      isMounted = false;
    };
  }, []);

  const formatMoney = (amount: number | string, currency = 'MYR') => {
    const numericAmount = Number(amount);

    if (!Number.isFinite(numericAmount)) {
      return `${currency} ${amount}`;
    }

    if (currency === 'MYR') {
      return `RM ${numericAmount.toLocaleString('en-MY', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`;
    }

    return `${currency} ${numericAmount.toLocaleString('en-MY', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  };

  if (isLoading) {
    return (
      <div className="p-6 md:p-8">
        <h1 className="text-2xl font-bold mb-1">Admin Dashboard</h1>
        <p className="text-muted-foreground mb-8">
          System overview and analytics.
        </p>

        <div className="rounded-xl border bg-card p-8 card-shadow text-center">
          <p className="text-sm text-muted-foreground">
            Loading system analytics...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8">
      <h1 className="text-2xl font-bold mb-1">Admin Dashboard</h1>
      <p className="text-muted-foreground mb-8">
        System overview and analytics.
      </p>

      {errorMessage && (
        <div
          className="mb-6 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
          role="alert"
        >
          {errorMessage}
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 mb-8">
        <StatCard
          title="Total Users"
          value={dashboard.total_users}
          icon={Users}
          description="registered user accounts"
        />

        <StatCard
          title="Total Predictions"
          value={dashboard.total_predictions}
          icon={Calculator}
          description="saved predictions"
        />

        <StatCard
          title="Public Predictions"
          value={`${dashboard.public_percentage.toFixed(1)}%`}
          icon={Building2}
          description="of saved predictions"
        />

        <StatCard
          title="Private Predictions"
          value={`${dashboard.private_percentage.toFixed(1)}%`}
          icon={Building}
          description="of saved predictions"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border bg-card p-5 card-shadow">
          <h3 className="font-semibold mb-4">Recent Users</h3>

          {dashboard.recent_users.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4">
              No registered users yet.
            </p>
          ) : (
            <div className="space-y-3">
              {dashboard.recent_users.map((user) => (
                <div
                  key={user.id}
                  className="flex items-center justify-between gap-4 text-sm"
                >
                  <div className="min-w-0">
                    <p className="font-medium truncate">
                      {user.full_name || 'Unnamed User'}
                    </p>
                    <p className="text-xs text-muted-foreground truncate">
                      {user.email || 'No email available'}
                    </p>
                  </div>

                  <span className="text-xs text-muted-foreground whitespace-nowrap">
                    {formatDateDisplay(user.created_at)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="rounded-xl border bg-card p-5 card-shadow">
          <h3 className="font-semibold mb-4">Recent Predictions</h3>

          {dashboard.recent_predictions.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4">
              No saved predictions yet.
            </p>
          ) : (
            <div className="space-y-3">
              {dashboard.recent_predictions.map((prediction) => (
                <div
                  key={prediction.id}
                  className="flex items-center justify-between gap-4 text-sm"
                >
                  <div className="min-w-0">
                    <p className="font-medium capitalize">
                      {prediction.hospital_type} Hospital
                    </p>
                    <p className="text-xs text-muted-foreground truncate">
                      {prediction.full_name ||
                        prediction.email ||
                        'Registered User'}{' '}
                      • {formatDateDisplay(prediction.created_at)}
                    </p>
                  </div>

                  <span className="font-semibold text-primary whitespace-nowrap">
                    {formatMoney(
                      prediction.total_cost,
                      prediction.currency,
                    )}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;