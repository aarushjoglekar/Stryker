import { Button } from "@/components/ui/button";
import { Play, Download, ArrowRight, Smartphone } from "lucide-react";

const CTA = () => {
  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 gradient-secondary opacity-90"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-maroon/20 rounded-full blur-3xl animate-pulse-glow"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Main CTA content */}
          <div className="animate-slide-up">
            <h2 className="text-4xl lg:text-6xl font-black mb-6 leading-tight">
              Ready to 
              <span className="text-gradient-primary"> Elevate </span>
              Your Game?
            </h2>
            <p className="text-xl lg:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto">
              Join the next generation of soccer analysis. Start your free trial today and 
              experience the power of AI-driven performance insights.
            </p>
          </div>

          {/* Action buttons */}
          <div className="flex flex-col lg:flex-row gap-6 justify-center items-center mb-16 animate-slide-up" style={{animationDelay: '0.2s'}}>
            <Button variant="hero" size="xl" className="group min-w-[200px]">
              <Play className="w-6 h-6 mr-3 group-hover:scale-110 transition-transform" />
              Watch Demo
              <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            </Button>
            
            <Button variant="glass" size="xl" className="min-w-[200px]">
              <Download className="w-6 h-6 mr-3" />
              Download App
            </Button>
          </div>

          {/* Feature highlights */}
          <div className="grid md:grid-cols-3 gap-8 mb-16 animate-slide-up" style={{animationDelay: '0.4s'}}>
            <div className="glass-card p-6 rounded-2xl text-center hover-lift">
              <div className="w-16 h-16 bg-gradient-to-br from-maroon to-maroon-light rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Play className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg mb-2">Free 14-Day Trial</h3>
              <p className="text-sm text-muted-foreground">
                Full access to all features with no credit card required
              </p>
            </div>

            <div className="glass-card p-6 rounded-2xl text-center hover-lift">
              <div className="w-16 h-16 bg-gradient-to-br from-navy to-navy-light rounded-2xl flex items-center justify-center mx-auto mb-4">
                <Smartphone className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg mb-2">Cross-Platform</h3>
              <p className="text-sm text-muted-foreground">
                Available on desktop, mobile, and tablet devices
              </p>
            </div>

            <div className="glass-card p-6 rounded-2xl text-center hover-lift">
              <div className="w-16 h-16 bg-gradient-to-br from-accent to-maroon-glow rounded-2xl flex items-center justify-center mx-auto mb-4">
                <ArrowRight className="w-8 h-8 text-white" />
              </div>
              <h3 className="font-bold text-lg mb-2">Instant Setup</h3>
              <p className="text-sm text-muted-foreground">
                Get started in minutes with our easy onboarding process
              </p>
            </div>
          </div>

          {/* Urgency/Social proof */}
          <div className="glass-card p-8 rounded-2xl animate-slide-up" style={{animationDelay: '0.6s'}}>
            <div className="flex flex-col lg:flex-row items-center justify-between gap-6">
              <div className="text-left">
                <h4 className="text-2xl font-bold text-gradient-primary mb-2">
                  Limited Time: Pro Features Free
                </h4>
                <p className="text-muted-foreground">
                  First 1,000 users get premium analytics and AI insights at no cost
                </p>
              </div>
              <div className="flex flex-col items-center lg:items-end">
                <div className="text-3xl font-black text-maroon mb-2">
                  847
                </div>
                <div className="text-sm text-muted-foreground">
                  spots remaining
                </div>
              </div>
            </div>
          </div>

          {/* Trust indicators */}
          <div className="flex flex-wrap justify-center items-center gap-8 mt-12 opacity-60 animate-slide-up" style={{animationDelay: '0.8s'}}>
            <div className="text-sm font-semibold">Trusted by:</div>
            <div className="flex gap-6 text-sm">
              <span>Premier League Teams</span>
              <span>•</span>
              <span>Olympic Athletes</span>
              <span>•</span>
              <span>Youth Academies</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTA;