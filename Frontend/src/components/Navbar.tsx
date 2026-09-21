import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import {
  Menu,
  X,
  Activity,
  User,
  LogIn,
  LogOut,
  LayoutDashboard,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { supabase } from '@/lib/supabase';

const Navbar = () => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [userEmail, setUserEmail] = useState('');
  const [isAuthLoading, setIsAuthLoading] = useState(true);
  const [isSigningOut, setIsSigningOut] = useState(false);

  const location = useLocation();
  const navigate = useNavigate();
  const isAdmin = location.pathname.startsWith('/admin');
  const isAuthenticated = Boolean(userEmail);

  useEffect(() => {
    let isMounted = true;

    const loadSession = async () => {
      const {
        data: { session },
        error,
      } = await supabase.auth.getSession();

      if (error) {
        console.error('Unable to read Supabase session:', error);
      }

      if (isMounted) {
        setUserEmail(session?.user.email ?? '');
        setIsAuthLoading(false);
      }
    };

    void loadSession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (isMounted) {
        setUserEmail(session?.user.email ?? '');
        setIsAuthLoading(false);
      }
    });

    return () => {
      isMounted = false;
      subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    setMobileOpen(false);
  }, [location.pathname]);

  if (isAdmin) return null;

  const navLinks = [
    { to: '/', label: 'Home' },
    { to: '/predict', label: 'Research & Pricing' },
  ];

  const handleLogout = async () => {
    setIsSigningOut(true);

    try {
      const { error } = await supabase.auth.signOut();

      if (error) {
        throw error;
      }

      setMobileOpen(false);
      navigate('/', { replace: true });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsSigningOut(false);
    }
  };

  return (
    <header className="sticky top-0 z-50 border-b bg-card/80 backdrop-blur-md">
      <div className="container flex h-16 items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-lg">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
            <Activity className="h-5 w-5 text-primary-foreground" />
          </div>
          <span className="hidden sm:inline">
            MyCare<span className="text-primary">Cost</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.to}
              to={link.to}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                location.pathname === link.to
                  ? 'bg-primary text-primary-foreground shadow-sm hover:bg-primary-hover'
                  : 'text-muted-foreground hover:bg-accent hover:text-primary'
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Desktop account actions */}
        <div className="hidden md:flex items-center gap-2">
          {isAuthLoading ? (
            <span className="text-sm text-muted-foreground">
              Checking account...
            </span>
          ) : isAuthenticated ? (
            <>
              <span
                className="max-w-[220px] truncate text-sm text-muted-foreground"
                title={userEmail}
              >
                {userEmail}
              </span>

              <Button variant="ghost" size="sm" asChild>
                <Link to="/dashboard">
                  <LayoutDashboard className="mr-2 h-4 w-4" />
                  Dashboard
                </Link>
              </Button>

              <Button
                variant="ghost"
                size="sm"
                type="button"
                onClick={handleLogout}
                disabled={isSigningOut}
              >
                <LogOut className="mr-2 h-4 w-4" />
                {isSigningOut ? 'Signing Out...' : 'Logout'}
              </Button>
            </>
          ) : (
            <>
              <Button variant="ghost" size="sm" asChild>
                <Link to="/login">
                  <LogIn className="mr-2 h-4 w-4" />
                  Login
                </Link>
              </Button>

              <Button size="sm" asChild>
                <Link to="/register">
                  <User className="mr-2 h-4 w-4" />
                  Register
                </Link>
              </Button>
            </>
          )}
        </div>

        {/* Mobile toggle */}
        <button
          type="button"
          className="md:hidden p-2"
          onClick={() => setMobileOpen((open) => !open)}
          aria-label={mobileOpen ? 'Close navigation menu' : 'Open navigation menu'}
        >
          {mobileOpen ? (
            <X className="h-5 w-5" />
          ) : (
            <Menu className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <div className="md:hidden border-t bg-card px-4 pb-4 animate-fade-in">
          <nav className="flex flex-col gap-1 pt-2">
            {navLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`px-4 py-2.5 rounded-md text-sm font-medium ${
                  location.pathname === link.to
                    ? 'bg-primary text-primary-foreground shadow-sm hover:bg-primary-hover'
                    : 'text-muted-foreground hover:bg-accent hover:text-primary'
                }`}
              >
                {link.label}
              </Link>
            ))}

            <hr className="my-2 border-border" />

            {isAuthLoading ? (
              <span className="px-4 py-2.5 text-sm text-muted-foreground">
                Checking account...
              </span>
            ) : isAuthenticated ? (
              <>
                <span
                  className="px-4 py-2 text-xs text-muted-foreground truncate"
                  title={userEmail}
                >
                  {userEmail}
                </span>

                <Link
                  to="/dashboard"
                  className="px-4 py-2.5 text-sm font-medium text-primary flex items-center gap-2"
                >
                  <LayoutDashboard className="h-4 w-4" />
                  Dashboard
                </Link>

                <button
                  type="button"
                  onClick={handleLogout}
                  disabled={isSigningOut}
                  className="px-4 py-2.5 text-left text-sm font-medium text-muted-foreground flex items-center gap-2 disabled:opacity-50"
                >
                  <LogOut className="h-4 w-4" />
                  {isSigningOut ? 'Signing Out...' : 'Logout'}
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="px-4 py-2.5 text-sm font-medium text-muted-foreground"
                >
                  Login
                </Link>
                <Link
                  to="/register"
                  className="px-4 py-2.5 text-sm font-medium text-primary"
                >
                  Register
                </Link>
              </>
            )}
          </nav>
        </div>
      )}
    </header>
  );
};

export default Navbar;
