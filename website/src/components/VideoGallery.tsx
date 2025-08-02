import React from 'react';
import { Play, Eye, Clock, Target } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface SampleVideo {
  id: string;
  title: string;
  thumbnail: string;
  videoPath: string
  duration: number;
  description: string;
}

interface VideoGalleryProps {
  onVideoSelect: (video: SampleVideo) => void;
}

const VideoGallery: React.FC<VideoGalleryProps> = ({ onVideoSelect }) => {
  const sampleVideos: SampleVideo[] = [
    {
      id: '1',
      title: 'PSG vs Nantes',
      thumbnail: 'videos/PSGNantes/PSGNantes.png',
      videoPath: 'videos/PSGNantes/PSGNantes.mp4',
      duration: 62,
      description: 'Ligue 1 League Match 2015',
    },
    {
      id: '2',
      title: 'Real Madrid vs Espanyol',
      thumbnail: 'videos/RealMadridEspanyol/RealMadridEspanyol.png',
      videoPath: 'videos/RealMadridEspanyol/RealMadridEspanyol.mp4',
      duration: 61,
      description: 'La Liga League Match 2016',
    },
    {
      id: '3',
      title: 'Dortmund vs Darmstadt',
      thumbnail: 'videos/DortmundDarmstadt/DortmundDarmstadt.png',
      videoPath: 'videos/DortmundDarmstadt/DortmundDarmstadt.mp4',
      duration: 62,
      description: 'Bundesliga League Match 2016',
    },
    {
      id: '4',
      title: 'PSG vs Ludogorets',
      thumbnail: 'videos/PSGLudogorets/PSGLudogorets.png',
      videoPath: 'videos/PSGLudogorets/PSGLudogorets.mp4',
      duration: 64,
      description: 'Champions League Group Stage 2016',
    },
    {
      id: '5',
      title: 'Argentina vs France',
      thumbnail: '/videos/FranceArgentina/FranceArgentina.png',
      videoPath: '/videos/FranceArgentina/FranceArgentina.mp4',
      duration: 60,
      description: 'World Cup Final 2022',
    },
    {
      id: '6',
      title: 'FC Barcelona vs Real Madrid',
      thumbnail: '/videos/BarcelonaRealMadrid/BarcelonaRealMadrid.png',
      videoPath: '/videos/BarcelonaRealMadrid/BarcelonaRealMadrid.mp4',
      duration: 72,
      description: 'El Classico 2017',
    },
  ];

  return (
    <div className="w-full">
      <div className="mb-8 text-center">
        <h2 className="text-3xl lg:text-4xl font-black mb-4">
          <span className="text-gradient-primary">OR</span>
        </h2>
        <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
          Choose from pre-uploaded soccer footage
        </p>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
        {sampleVideos.map((video, index) => {

          return (
            <div
              key={video.id}
              className="group glass-card rounded-2xl overflow-hidden hover-lift animate-slide-up"
              style={{ animationDelay: `${index * 0.1}s` }}
            >

              {/* Video thumbnail */}
              <div className="relative overflow-hidden">
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  className="aspect-video w-full object-cover transition-transform duration-300 group-hover:scale-105"
                />

                {/* Overlay */}
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <Button
                    variant="hero"
                    size="lg"
                    onClick={() => onVideoSelect(video)}
                    className="transform scale-90 group-hover:scale-100 transition-transform"
                  >
                    <Play className="w-5 h-5 mr-2" />
                    Analyze Video
                  </Button>
                </div>

                {/* Duration badge */}
                <div className="absolute top-3 right-3 glass-card px-2 py-1 rounded-lg">
                  <span className="text-xs font-semibold text-foreground">{video.duration}s</span>
                </div>

                {/* Type badge */}
                <div className="absolute top-3 left-3 bg-gradient-to-r from-navy to-navy-light px-3 py-1 rounded-full">
                  <div className="flex items-center gap-1">
                    <Eye className="w-3 h-3" />
                    <span className="text-xs font-semibold text-white capitalize">AI Ready</span>
                  </div>
                </div>
              </div>


              {/* Content */}
              <div className="p-6">
                <h3 className="font-bold text-lg mb-2 text-foreground group-hover:text-gradient-primary transition-colors">
                  {video.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed mb-4">
                  {video.description}
                </p>
              </div>

              {/* Hover glow effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-maroon/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none rounded-2xl"></div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default VideoGallery;