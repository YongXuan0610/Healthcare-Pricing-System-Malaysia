import { ReactNode, useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Shield,
  LayoutDashboard,
  Users,
  DollarSign,
  Settings,
  LogOut,
} from 'lucide-react';
import { supabase } from '@/lib/supabase';

const navItems = [
  { to: '/admin', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/admin/users', label: 'Users', icon: Users },
  { to: '/admin/pricing', label: 'Pricing', icon: DollarSign },
  { to: '/admin/settings', label: 'Settings', icon: Settings },
];

const AdminLayout = ({ children }: { children: ReactNode }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const [isAuthorised, setIsAuthorised] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const verifyAdminAccess = async () => {
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
          navigate('/admin/login', { replace: true });
          return;
        }

        if (userError) {
          throw userError;
        }

        const { data: profile, error: profileError } = await supabase
          .from('profiles')
          .select('role')
          .eq('id', user.id)
          .maybeSingle();

        if (profileError) {
          throw profileError;
        }

        if (profile?.role !== 'admin') {
          isRedirecting = true;
          navigate('/dashboard', { replace: true });
          return;
        }

        if (isMounted) {
          setIsAuthorised(true);
        }
      } catch (error) {
        console.error('Admin access verification error:', error);

        if (isMounted) {
          setErrorMessage(
            'Unable to verify administrator access. Please sign in again.',
          );
          setIsAuthorised(false);
        }
      } finally {
        if (isMounted && !isRedirecting) {
          setIsLoading(false);
        }
      }
    };

    void verifyAdminAccess();

    return () => {
      isMounted = false;
    };
  }, [navigate, location.pathname]);

  const handleExitAdmin = async () => {
    setIsSigningOut(true);
    setErrorMessage('');

    try {
      const { error } = await supabase.auth.signOut();

      if (error) {
        throw error;
      }

      navigate('/', { replace: true });
    } catch (error) {
      console.error('Admin sign-out error:', error);
      setErrorMessage('Unable to sign out. Please try again.');
    } finally {
      setIsSigningOut(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-muted/30">
        <div className="text-center">
          <Shield className="h-8 w-8 text-primary mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">
            Verifying administrator access...
          </p>
        </div>
      </div>
    );
  }

  if (!isAuthorised) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-muted/30 px-4">
        <div className="text-center max-w-md">
          <Shield className="h-8 w-8 text-destructive mx-auto mb-3" />
          <p className="font-semibold mb-1">Administrator access unavailable</p>
          <p className="text-sm text-muted-foreground">
            {errorMessage || 'You do not have permission to access this page.'}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex bg-muted/30">
      {/* Sidebar */}
      <aside className="hidden md:flex w-64 flex-col bg-card border-r">
        <div className="p-5 border-b">
          <div className="flex items-center gap-2 font-bold text-lg">
            <Shield className="h-5 w-5 text-primary" />
            Admin Panel
          </div>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {navItems.map((item) => {
            const active = location.pathname === item.to;

            return (
              <Link
                key={item.to}
                to={item.to}
                className={`flex items-center gap-3 rounded-md px-3 py-2.5 text-sm font-medium transition-colors ${
                  active
                    ? 'bg-primary text-primary-foreground shadow-sm hover:bg-primary-hover'
                    : 'text-muted-foreground hover:bg-accent hover:text-primary'
                }`}
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="p-3 border-t">
          {errorMessage && (
            <p className="px-3 pb-2 text-xs text-destructive" role="alert">
              {errorMessage}
            </p>
          )}

          <button
            type="button"
            onClick={handleExitAdmin}
            disabled={isSigningOut}
            className="w-full flex items-center gap-3 rounded-md px-3 py-2.5 text-sm text-muted-foreground hover:bg-muted transition-colors disabled:opacity-50"
          >
            <LogOut className="h-4 w-4" />
            {isSigningOut ? 'Signing Out...' : 'Exit Admin'}
          </button>
        </div>
      </aside>

      {/* Mobile top bar */}
      <div className="flex-1 flex flex-col">
        <header className="md:hidden sticky top-0 z-50 border-b bg-card/80 backdrop-blur-md">
          <div className="flex h-14 items-center justify-between px-4">
            <div className="flex items-center gap-2 font-bold text-sm">
              <Shield className="h-4 w-4 text-primary" />
              Admin
            </div>

            <div className="flex gap-1">
              {navItems.map((item) => {
                const active = location.pathname === item.to;

                return (
                  <Link
                    key={item.to}
                    to={item.to}
                    className={`p-2 rounded-md ${
                      active
                        ? 'bg-primary text-primary-foreground shadow-sm hover:bg-primary-hover'
                        : 'text-muted-foreground hover:bg-accent hover:text-primary'
                    }`}
                    aria-label={item.label}
                    title={item.label}
                  >
                    <item.icon className="h-4 w-4" />
                  </Link>
                );
              })}

              <button
                type="button"
                onClick={handleExitAdmin}
                disabled={isSigningOut}
                className="p-2 rounded-md text-muted-foreground disabled:opacity-50"
                aria-label="Exit admin"
                title="Exit admin"
              >
                <LogOut className="h-4 w-4" />
              </button>
            </div>
          </div>
        </header>

        {errorMessage && (
          <div
            className="md:hidden border-b border-destructive/20 bg-destructive/5 px-4 py-2 text-xs text-destructive"
            role="alert"
          >
            {errorMessage}
          </div>
        )}

        <main className="flex-1">{children}</main>
      </div>
    </div>
  );
};

export default AdminLayout;
