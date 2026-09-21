import { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: LucideIcon;
  description?: string;
  trend?: string;
}

const StatCard = ({ title, value, icon: Icon, description, trend }: StatCardProps) => (
  <div className="rounded-lg border bg-card p-5 card-shadow">
    <div className="flex items-center justify-between mb-3">
      <p className="text-sm text-muted-foreground">{title}</p>
      <div className="h-9 w-9 rounded-md bg-accent flex items-center justify-center">
        <Icon className="h-4 w-4 text-accent-foreground" />
      </div>
    </div>
    <p className="text-2xl font-bold">{value}</p>
    {(description || trend) && (
      <p className="text-xs text-muted-foreground mt-1">{trend && <span className="text-healthcare-green font-medium">{trend} </span>}{description}</p>
    )}
  </div>
);

export default StatCard;
