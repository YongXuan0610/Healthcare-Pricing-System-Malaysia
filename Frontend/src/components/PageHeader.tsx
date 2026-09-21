interface PageHeaderProps {
  title: string;
  description?: string;
  gradient?: boolean;
}

const PageHeader = ({ title, description, gradient }: PageHeaderProps) => (
  <div className={`py-10 ${gradient ? 'healthcare-gradient text-primary-foreground' : 'bg-muted/50'}`}>
    <div className="container">
      <h1 className="text-2xl md:text-3xl font-bold">{title}</h1>
      {description && <p className={`mt-2 text-sm md:text-base ${gradient ? 'text-primary-foreground/80' : 'text-muted-foreground'}`}>{description}</p>}
    </div>
  </div>
);

export default PageHeader;
