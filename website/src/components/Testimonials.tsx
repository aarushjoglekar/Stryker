import { Star, Quote } from "lucide-react";
import playerAction from "@/assets/player-action.jpg";
import coachAnalysis from "@/assets/coach-analysis.jpg";

const Testimonials = () => {
  const testimonials = [
    {
      name: "Marcus Rodriguez",
      role: "Professional Player",
      team: "Premier League",
      image: playerAction,
      rating: 5,
      quote: "This platform transformed my understanding of the game. The AI analysis helped me improve my positioning by 40% in just three months.",
      stats: "40% improvement in positioning"
    },
    {
      name: "Sarah Johnson",
      role: "Head Coach",
      team: "Elite Soccer Academy",
      image: coachAnalysis,
      rating: 5,
      quote: "The tactical insights are incredible. We can now analyze opponent patterns and adjust our strategy in real-time during matches.",
      stats: "85% win rate improvement"
    },
    {
      name: "David Chen",
      role: "Performance Analyst",
      team: "National Team",
      image: playerAction,
      rating: 5,
      quote: "The depth of analytics is unmatched. Heat maps, movement patterns, and performance metrics all in one comprehensive platform.",
      stats: "300+ players analyzed"
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-navy/10 to-transparent"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-maroon/10 rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Section header */}
        <div className="text-center mb-16 animate-slide-up">
          <h2 className="text-4xl lg:text-5xl font-black mb-6">
            <span className="text-gradient-primary">Trusted by Champions</span>
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            From grassroots to professional level, our platform is empowering players and coaches worldwide 
            to achieve their highest potential.
          </p>
        </div>

        {/* Testimonials grid */}
        <div className="grid lg:grid-cols-3 gap-8 mb-16">
          {testimonials.map((testimonial, index) => (
            <div 
              key={index}
              className="group glass-card p-8 rounded-2xl hover-lift animate-slide-up relative overflow-hidden"
              style={{animationDelay: `${index * 0.2}s`}}
            >
              {/* Quote icon */}
              <div className="absolute top-6 right-6 opacity-20">
                <Quote className="w-12 h-12 text-maroon" />
              </div>

              {/* Profile section */}
              <div className="flex items-center gap-4 mb-6">
                <div className="relative">
                  <img 
                    src={testimonial.image} 
                    alt={testimonial.name}
                    className="w-16 h-16 rounded-full object-cover border-2 border-maroon/30"
                  />
                  <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-maroon rounded-full flex items-center justify-center">
                    <Star className="w-3 h-3 text-white fill-current" />
                  </div>
                </div>
                <div>
                  <h4 className="font-bold text-foreground">{testimonial.name}</h4>
                  <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                  <p className="text-sm text-maroon font-semibold">{testimonial.team}</p>
                </div>
              </div>

              {/* Rating */}
              <div className="flex gap-1 mb-4">
                {[...Array(testimonial.rating)].map((_, i) => (
                  <Star key={i} className="w-4 h-4 text-maroon fill-current" />
                ))}
              </div>

              {/* Quote */}
              <p className="text-foreground leading-relaxed mb-6 italic">
                "{testimonial.quote}"
              </p>

              {/* Stats highlight */}
              <div className="glass-card p-3 rounded-lg border border-maroon/20">
                <p className="text-sm text-maroon font-semibold text-center">
                  {testimonial.stats}
                </p>
              </div>

              {/* Hover effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-maroon/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-2xl"></div>
            </div>
          ))}
        </div>

        {/* Stats section */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 animate-slide-up" style={{animationDelay: '0.6s'}}>
          {[
            { number: "10k+", label: "Active Users" },
            { number: "500+", label: "Teams" },
            { number: "99.9%", label: "Uptime" },
            { number: "50+", label: "Countries" }
          ].map((stat, index) => (
            <div key={index} className="text-center glass-card p-6 rounded-xl hover-glow">
              <div className="text-3xl lg:text-4xl font-black text-gradient-primary mb-2">
                {stat.number}
              </div>
              <div className="text-muted-foreground font-medium">
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Testimonials;