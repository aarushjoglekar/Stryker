import { Brain, Video, TrendingUp, Users, Timer, Award } from "lucide-react";

const Features = () => {
  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms analyze player movements, tactics, and performance metrics in real-time.",
      gradient: "from-maroon to-maroon-light"
    },
    {
      icon: Video,
      title: "Smart Highlights",
      description: "Automatically generates highlight reels of key moments, goals, saves, and tactical plays.",
      gradient: "from-accent to-maroon-glow"
    },
    {
      icon: TrendingUp,
      title: "Performance Metrics",
      description: "Track detailed statistics including speed, distance, pass accuracy, and heat maps.",
      gradient: "from-navy to-navy-light"
    },
    {
      icon: Users,
      title: "Team Analytics",
      description: "Comprehensive team formation analysis, player positioning, and tactical insights.",
      gradient: "from-maroon-light to-accent"
    },
    {
      icon: Timer,
      title: "Real-time Processing",
      description: "Get instant feedback during matches with live performance monitoring and alerts.",
      gradient: "from-navy-light to-maroon"
    },
    {
      icon: Award,
      title: "Progress Tracking",
      description: "Monitor improvement over time with detailed progress reports and skill development insights.",
      gradient: "from-accent to-navy-light"
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-20 left-20 w-96 h-96 bg-maroon rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-20 w-96 h-96 bg-navy rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Section header */}
        <div className="text-center mb-16 animate-slide-up">
          <h2 className="text-4xl lg:text-5xl font-black mb-6">
            <span className="text-gradient-primary">Revolutionary Features</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Cutting-edge technology meets soccer intelligence. Our platform transforms how you analyze, 
            understand, and improve your game performance.
          </p>
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="group glass-card p-8 rounded-2xl hover-lift hover-glow animate-slide-up"
              style={{animationDelay: `${index * 0.1}s`}}
            >
              {/* Icon with gradient background */}
              <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${feature.gradient} mb-6 group-hover:scale-110 transition-transform duration-300`}>
                <feature.icon className="w-8 h-8 text-white" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-bold mb-4 text-foreground group-hover:text-gradient-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>

              {/* Hover effect overlay */}
              <div className="absolute inset-0 bg-gradient-to-br from-maroon/5 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
            </div>
          ))}
        </div>

        {/* Bottom CTA */}
        <div className="text-center mt-16 animate-slide-up" style={{animationDelay: '0.8s'}}>
          <div className="glass-card inline-block p-8 rounded-2xl">
            <h3 className="text-2xl font-bold mb-4 text-gradient-primary">
              Ready to revolutionize your game?
            </h3>
            <p className="text-muted-foreground mb-6">
              Join thousands of players and coaches already using our platform.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="gradient-primary text-white px-8 py-3 rounded-lg font-semibold hover:scale-105 glow-maroon transition-all duration-300">
                Start Free Trial
              </button>
              <button className="border-2 border-maroon bg-transparent text-maroon px-8 py-3 rounded-lg font-semibold hover:bg-maroon hover:text-white transition-all duration-300">
                View Pricing
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;