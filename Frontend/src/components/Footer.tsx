import { Activity } from 'lucide-react';
import { Link } from 'react-router-dom';

const Footer = () => (
  <footer className="border-t bg-card mt-auto">
    <div className="container py-8">
      <div className="grid gap-8 md:grid-cols-3">
        <div>
          <Link to="/" className="flex items-center gap-2 font-bold text-lg mb-3">
            <Activity className="h-5 w-5 text-primary" />
            MyCare<span className="text-primary">Cost</span>
          </Link>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Healthcare cost research with an NSS 80 primary model, an isolated US benchmark, and separate Malaysian pricing references.
          </p>
        </div>
        <div>
          <h4 className="font-semibold mb-3 text-sm">Quick Links</h4>
          <div className="flex flex-col gap-2 text-sm text-muted-foreground">
            <Link to="/predict" className="hover:text-foreground transition-colors">Research & Pricing</Link>
            <Link to="/login" className="hover:text-foreground transition-colors">Login</Link>
            <Link to="/register" className="hover:text-foreground transition-colors">Register</Link>
          </div>
        </div>
        <div>
          <h4 className="font-semibold mb-3 text-sm">Disclaimer</h4>
          <p className="text-sm text-muted-foreground leading-relaxed">
            Research estimates and published price references are not medical advice, insurance decisions, or guaranteed bills.
          </p>
        </div>
      </div>
      <div className="border-t mt-6 pt-6 text-center text-xs text-muted-foreground">
        © 2026 MyCareCost. Final Year Project — Healthcare Cost Research System.
      </div>
    </div>
  </footer>
);

export default Footer;
