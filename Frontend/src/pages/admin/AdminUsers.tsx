import { useEffect, useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Search, Trash2, Eye, Shield, UserRound } from 'lucide-react';
import { formatDateDisplay } from '@/lib/utils';
import { supabase } from '@/lib/supabase';

type AdminUserSummary = {
  id: string;
  full_name: string;
  email: string | null;
  state: string | null;
  role: 'user' | 'admin';
  created_at: string;
  predictions_count: number | string;
};

const AdminUsers = () => {
  const [search, setSearch] = useState('');
  const [users, setUsers] = useState<AdminUserSummary[]>([]);
  const [selectedUser, setSelectedUser] = useState<AdminUserSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const loadUsers = async () => {
    setIsLoading(true);
    setErrorMessage('');

    try {
      const { data, error } = await supabase.rpc('get_admin_user_summaries');

      if (error) {
        throw error;
      }

      setUsers((data ?? []) as AdminUserSummary[]);
    } catch (error) {
      console.error('Admin user loading error:', error);
      setErrorMessage('Unable to load registered users.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void loadUsers();
  }, []);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) {
      return users;
    }

    return users.filter((user) =>
      [user.full_name, user.email ?? '', user.state ?? '', user.role]
        .some((value) => value.toLowerCase().includes(query)),
    );
  }, [search, users]);

  const handleDelete = async (user: AdminUserSummary) => {
    if (user.role === 'admin') {
      setErrorMessage('Administrator accounts cannot be deleted from this page.');
      return;
    }

    const confirmed = window.confirm(
      `Delete ${user.full_name || user.email || 'this user'}? This will also remove the user's saved prediction history.`,
    );

    if (!confirmed) {
      return;
    }

    setDeletingId(user.id);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const { error } = await supabase.rpc('delete_user_as_admin', {
        target_user_id: user.id,
      });

      if (error) {
        throw error;
      }

      setUsers((current) => current.filter((item) => item.id !== user.id));
      setSelectedUser(null);
      setSuccessMessage('User account deleted successfully.');
    } catch (error) {
      console.error('Admin user deletion error:', error);
      setErrorMessage('Unable to delete this user account.');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="p-6 md:p-8">
      <h1 className="text-2xl font-bold mb-1">User Management</h1>
      <p className="text-muted-foreground mb-6">
        View registered users and their prediction activity.
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

      <div className="flex items-center gap-3 mb-6 max-w-sm">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search users..."
            className="pl-9"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
      </div>

      <div className="rounded-xl border bg-card card-shadow overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                  Name
                </th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden sm:table-cell">
                  Email
                </th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden md:table-cell">
                  State
                </th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground">
                  Role
                </th>
                <th className="text-left px-4 py-3 font-medium text-muted-foreground hidden lg:table-cell">
                  Joined
                </th>
                <th className="text-center px-4 py-3 font-medium text-muted-foreground">
                  Predictions
                </th>
                <th className="text-right px-4 py-3 font-medium text-muted-foreground">
                  Actions
                </th>
              </tr>
            </thead>

            <tbody>
              {isLoading ? (
                <tr>
                  <td
                    colSpan={7}
                    className="px-4 py-10 text-center text-muted-foreground"
                  >
                    Loading registered users...
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td
                    colSpan={7}
                    className="px-4 py-10 text-center text-muted-foreground"
                  >
                    No users found.
                  </td>
                </tr>
              ) : (
                filtered.map((user) => (
                  <tr
                    key={user.id}
                    className="border-b last:border-0 hover:bg-muted/30 transition-colors"
                  >
                    <td className="px-4 py-3 font-medium">
                      {user.full_name || 'Unnamed User'}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground hidden sm:table-cell">
                      {user.email || '—'}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground hidden md:table-cell">
                      {user.state || '—'}
                    </td>
                    <td className="px-4 py-3">
                      <span
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${
                          user.role === 'admin'
                            ? 'bg-primary/10 text-primary'
                            : 'bg-muted text-muted-foreground'
                        }`}
                      >
                        {user.role === 'admin' ? (
                          <Shield className="h-3 w-3" />
                        ) : (
                          <UserRound className="h-3 w-3" />
                        )}
                        {user.role}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-muted-foreground hidden lg:table-cell">
                      {formatDateDisplay(user.created_at)}
                    </td>
                    <td className="px-4 py-3 text-center">
                      {Number(user.predictions_count).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div className="flex justify-end gap-1">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={() => setSelectedUser(user)}
                          aria-label={`View ${user.full_name}`}
                          title="View user"
                        >
                          <Eye className="h-3.5 w-3.5" />
                        </Button>

                        <Button
                          type="button"
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-destructive"
                          onClick={() => handleDelete(user)}
                          disabled={
                            user.role === 'admin' || deletingId === user.id
                          }
                          aria-label={`Delete ${user.full_name}`}
                          title={
                            user.role === 'admin'
                              ? 'Administrator accounts cannot be deleted here'
                              : 'Delete user'
                          }
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      <Dialog
        open={Boolean(selectedUser)}
        onOpenChange={(open) => {
          if (!open) {
            setSelectedUser(null);
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>User Details</DialogTitle>
            <DialogDescription>
              Registered account information and system activity.
            </DialogDescription>
          </DialogHeader>

          {selectedUser && (
            <div className="grid gap-3 text-sm">
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Full Name</span>
                <span className="font-medium text-right">
                  {selectedUser.full_name || '—'}
                </span>
              </div>

              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Email</span>
                <span className="font-medium text-right">
                  {selectedUser.email || '—'}
                </span>
              </div>

              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">State</span>
                <span className="font-medium text-right">
                  {selectedUser.state || '—'}
                </span>
              </div>

              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Role</span>
                <span className="font-medium capitalize">
                  {selectedUser.role}
                </span>
              </div>

              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Joined</span>
                <span className="font-medium">
                  {formatDateDisplay(selectedUser.created_at)}
                </span>
              </div>

              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">
                  Saved Predictions
                </span>
                <span className="font-medium">
                  {Number(selectedUser.predictions_count).toLocaleString()}
                </span>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default AdminUsers;
