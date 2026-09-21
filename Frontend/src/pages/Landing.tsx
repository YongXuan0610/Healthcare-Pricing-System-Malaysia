import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import {
  Activity,
  Building2,
  BookOpen,
  Database,
  FlaskConical,
  TrendingUp,
} from 'lucide-react';

const Landing = () => {
  const features = [
    {
      icon: FlaskConical,
      title: 'NSS 80 Primary Model',
      desc: 'Research estimates for Indian inpatient medical expenditure using the official 2025 survey.',
    },
    {
      icon: Database,
      title: 'US Benchmark',
      desc: 'The 1,338-row Kaggle model remains available only as a secondary USD benchmark.',
    },
    {
      icon: BookOpen,
      title: 'LIAM Reference',
      desc: 'Published Malaysian private-healthcare P25, typical, and P75 bills stay non-ML.',
    },
    {
      icon: Building2,
      title: 'Hospital References',
      desc: 'Malaysian public and private source-supported pricing remains in separate workflows.',
    },
  ];

  return (
    <div className="flex flex-col">
      {/* Hero */}
      <section className="healthcare-gradient text-primary-foreground py-20 md:py-28">
        <div className="container">
          <div className="max-w-2xl animate-fade-in">
            <div className="inline-flex items-center gap-2 rounded-full bg-primary-foreground/15 px-4 py-1.5 text-sm font-medium mb-6">
              <Activity className="h-4 w-4" />
              Clearer Healthcare Costs. Smarter Planning.
            </div>

            <h1 className="text-3xl md:text-5xl font-extrabold leading-tight mb-4">
              Know Your Healthcare Cost Before You Decide
            </h1>

            <p className="text-lg text-primary-foreground/80 mb-8 leading-relaxed max-w-xl">
              Plan with more confidence by exploring public and private
              healthcare prices, individual cost predictions, and trusted
              Malaysian pricing references — all presented clearly in one
              easy-to-use platform.
            </p>

            <div>
              <Button size="lg" variant="secondary" asChild>
                <Link to="/predict">Cost &amp; Pricing</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 md:py-20">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-bold mb-3">
              Three Roles, Kept Separate
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Each result retains its source population, target definition, and
              currency. No NSS, Kaggle, LIAM, or hospital-pricing rows are
              merged.
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((feature, index) => (
              <div
                key={feature.title}
                className="rounded-xl border bg-card p-6 card-shadow hover:card-shadow-hover transition-shadow"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="h-11 w-11 rounded-lg bg-accent flex items-center justify-center mb-4">
                  <feature.icon className="h-5 w-5 text-accent-foreground" />
                </div>
                <h3 className="font-semibold mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 bg-muted/50">
        <div className="container text-center">
          <div className="flex items-center justify-center gap-2 mb-4">
            <TrendingUp className="h-6 w-6 text-primary" />
          </div>

          <h2 className="text-2xl md:text-3xl font-bold mb-3">
            Explore the Evidence
          </h2>

          <p className="text-muted-foreground mb-6 max-w-lg mx-auto">
            Choose the NSS research model, US benchmark, LIAM publication, or
            hospital pricing route that matches your question.
          </p>

          <Button size="lg" asChild>
            <Link to="/predict">Cost &amp; Pricing</Link>
          </Button>
        </div>
      </section>
    </div>
  );
};

export default Landing;
