import { Button } from "@/components/ui/button";
import { Play, BarChart3, Target } from "lucide-react";
import heroMockup from "@/assets/hero-mockup.jpg";

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Background gradient */}
      <div className="absolute inset-0 gradient-hero opacity-90"></div>
      
      {/* Animated background particles */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-maroon rounded-full animate-float"></div>
        <div className="absolute top-1/3 right-1/4 w-1 h-1 bg-maroon-light rounded-full animate-float" style={{animationDelay: '1s'}}></div>
        <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-accent rounded-full animate-float" style={{animationDelay: '2s'}}></div>
        <div className="absolute top-2/3 right-1/3 w-1 h-1 bg-maroon-glow rounded-full animate-float" style={{animationDelay: '0.5s'}}></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Hero Content */}
          <div className="space-y-8 animate-slide-up">
            <div className="space-y-4">
              <h1 className="text-5xl lg:text-7xl font-black leading-tight">
                <span className="text-gradient-primary">Analyze.</span>
                <br />
                <span className="text-foreground">Improve.</span>
                <br />
                <span className="text-gradient-primary">Dominate.</span>
              </h1>
              <p className="text-xl lg:text-2xl text-muted-foreground max-w-lg">
                Revolutionary AI-powered video analysis for soccer players, coaches, and analysts. 
                Turn every match into actionable insights.
              </p>
            </div>

            {/* Key stats */}
            <div className="flex flex-wrap gap-6 text-sm">
              <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
                <BarChart3 className="w-4 h-4 text-maroon" />
                <span>99% Accuracy</span>
              </div>
              <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
                <Target className="w-4 h-4 text-maroon" />
                <span>Real-time Analysis</span>
              </div>
              <div className="flex items-center gap-2 glass-card px-4 py-2 rounded-full">
                <Play className="w-4 h-4 text-maroon" />
                <span>AI-Powered</span>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button variant="hero" size="xl" className="group">
                <Play className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
                Watch Demo
              </Button>
              <Button variant="outline-glow" size="xl">
                Start Free Trial
              </Button>
            </div>
          </div>

          {/* Hero Image/Mockup */}
          <div className="relative animate-slide-up" style={{animationDelay: '0.3s'}}>
            <div className="relative">
              {/* Glowing border effect */}
              <div className="absolute -inset-4 gradient-primary rounded-2xl opacity-20 blur-xl animate-pulse-glow"></div>
              
              {/* Main mockup */}
              <div className="relative glass-card p-4 rounded-2xl">
                <img 
                  src={heroMockup} 
                  alt="Soccer Analysis App Interface"
                  className="w-full h-auto rounded-lg shadow-2xl"
                />
                
                {/* Floating UI elements */}
                <div className="absolute top-8 right-8 glass-card p-3 rounded-lg animate-float">
                  <div className="text-xs text-muted-foreground">Live Analysis</div>
                  <div className="text-lg font-bold text-maroon">94%</div>
                </div>
                
                <div className="absolute bottom-8 left-8 glass-card p-3 rounded-lg animate-float" style={{animationDelay: '1s'}}>
                  <div className="text-xs text-muted-foreground">Pass Accuracy</div>
                  <div className="text-lg font-bold text-accent">87%</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent"></div>
    </section>
  );
};

export default Hero;