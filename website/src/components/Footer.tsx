import { Twitter, Instagram, Youtube, Linkedin, Mail, MapPin, Phone } from "lucide-react";

const Footer = () => {
  const socialLinks = [
    { icon: Twitter, href: "#", label: "Twitter" },
    { icon: Instagram, href: "#", label: "Instagram" },
    { icon: Youtube, href: "#", label: "YouTube" },
    { icon: Linkedin, href: "#", label: "LinkedIn" }
  ];

  const footerLinks = {
    product: [
      { name: "Features", href: "#features" },
      { name: "Pricing", href: "#pricing" },
      { name: "Demo", href: "#demo" },
      { name: "API", href: "#api" }
    ],
    company: [
      { name: "About", href: "#about" },
      { name: "Careers", href: "#careers" },
      { name: "Press", href: "#press" },
      { name: "Blog", href: "#blog" }
    ],
    support: [
      { name: "Help Center", href: "#help" },
      { name: "Contact", href: "#contact" },
      { name: "Status", href: "#status" },
      { name: "Documentation", href: "#docs" }
    ],
    legal: [
      { name: "Privacy Policy", href: "#privacy" },
      { name: "Terms of Service", href: "#terms" },
      { name: "Cookie Policy", href: "#cookies" },
      { name: "GDPR", href: "#gdpr" }
    ]
  };

  return (
    <footer className="relative bg-navy border-t border-border/30">
      {/* Background pattern */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-0 left-1/4 w-64 h-64 bg-maroon rounded-full blur-3xl"></div>
        <div className="absolute bottom-0 right-1/4 w-64 h-64 bg-accent rounded-full blur-3xl"></div>
      </div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Main footer content */}
        <div className="py-16">
          <div className="grid lg:grid-cols-12 gap-12">
            {/* Brand section */}
            <div className="lg:col-span-4">
              <div className="mb-6">
                <h3 className="text-2xl font-black text-gradient-primary mb-4">
                  SoccerAnalytics AI
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  Revolutionizing soccer performance analysis with cutting-edge AI technology. 
                  Empowering players, coaches, and teams to reach their full potential.
                </p>
              </div>

              {/* Contact info */}
              <div className="space-y-3 text-sm">
                <div className="flex items-center gap-3 text-muted-foreground">
                  <Mail className="w-4 h-4 text-maroon" />
                  <span>hello@socceranalytics.ai</span>
                </div>
                <div className="flex items-center gap-3 text-muted-foreground">
                  <Phone className="w-4 h-4 text-maroon" />
                  <span>+1 (555) 123-4567</span>
                </div>
                <div className="flex items-center gap-3 text-muted-foreground">
                  <MapPin className="w-4 h-4 text-maroon" />
                  <span>San Francisco, CA</span>
                </div>
              </div>

              {/* Social links */}
              <div className="flex gap-4 mt-6">
                {socialLinks.map((social, index) => (
                  <a
                    key={index}
                    href={social.href}
                    aria-label={social.label}
                    className="w-10 h-10 glass-card rounded-lg flex items-center justify-center hover:bg-maroon/20 hover:border-maroon/50 transition-all duration-300 group"
                  >
                    <social.icon className="w-5 h-5 text-muted-foreground group-hover:text-maroon transition-colors" />
                  </a>
                ))}
              </div>
            </div>

            {/* Links sections */}
            <div className="lg:col-span-8">
              <div className="grid md:grid-cols-4 gap-8">
                <div>
                  <h4 className="font-bold text-foreground mb-4">Product</h4>
                  <ul className="space-y-3">
                    {footerLinks.product.map((link, index) => (
                      <li key={index}>
                        <a
                          href={link.href}
                          className="text-muted-foreground hover:text-maroon transition-colors duration-300"
                        >
                          {link.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-bold text-foreground mb-4">Company</h4>
                  <ul className="space-y-3">
                    {footerLinks.company.map((link, index) => (
                      <li key={index}>
                        <a
                          href={link.href}
                          className="text-muted-foreground hover:text-maroon transition-colors duration-300"
                        >
                          {link.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-bold text-foreground mb-4">Support</h4>
                  <ul className="space-y-3">
                    {footerLinks.support.map((link, index) => (
                      <li key={index}>
                        <a
                          href={link.href}
                          className="text-muted-foreground hover:text-maroon transition-colors duration-300"
                        >
                          {link.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>

                <div>
                  <h4 className="font-bold text-foreground mb-4">Legal</h4>
                  <ul className="space-y-3">
                    {footerLinks.legal.map((link, index) => (
                      <li key={index}>
                        <a
                          href={link.href}
                          className="text-muted-foreground hover:text-maroon transition-colors duration-300"
                        >
                          {link.name}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="border-t border-border/30 py-8">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="text-sm text-muted-foreground">
              © 2024 SoccerAnalytics AI. All rights reserved.
            </div>
            <div className="flex items-center gap-4 text-sm">
              <span className="text-muted-foreground">Made with</span>
              <span className="text-maroon">❤️</span>
              <span className="text-muted-foreground">for soccer lovers worldwide</span>
            </div>
          </div>
        </div>
      </div>

      {/* Glow effect at bottom */}
      <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-maroon to-transparent opacity-50"></div>
    </footer>
  );
};

export default Footer;