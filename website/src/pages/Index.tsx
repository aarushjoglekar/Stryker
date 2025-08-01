import React, { useState } from 'react';
import { Brain, Zap, Target, Sparkles } from 'lucide-react';
import VideoUpload from '@/components/VideoUpload';
import VideoGallery from '@/components/VideoGallery';
import VideoAnalysis from '@/components/VideoAnalysis';
import type { SampleVideo } from '@/components/VideoGallery';

const Index = () => {
  const [selectedVideo, setSelectedVideo] = useState<{ file?: File; id?: string; metadata?: SampleVideo } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleVideoUpload = async (file: File) => {
    setIsProcessing(true);
    setUploadProgress(0);

    // Simulate processing with progress updates
    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          setIsProcessing(false);
          setSelectedVideo({ file });
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 500);
  };

  const handleSampleVideoSelect = (video: SampleVideo) => {
    +  setSelectedVideo({ id: video.id, metadata: video });
  }

  const handleCloseAnalysis = () => {
    setSelectedVideo(null);
    setIsProcessing(false);
    setUploadProgress(0);
  };

  if (selectedVideo) {
    return (
      <VideoAnalysis
        videoPath={selectedVideo.metadata.videoPath}
        matchTitle={selectedVideo.metadata?.title}
        matchDescription={selectedVideo.metadata?.description}
        clipDuration={selectedVideo.metadata?.duration}
        onClose={handleCloseAnalysis}
      />

    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 gradient-hero opacity-40"></div>
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-maroon/10 rounded-full blur-3xl animate-pulse-glow"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-navy/10 rounded-full blur-3xl animate-pulse-glow" style={{ animationDelay: '1s' }}></div>
      </div>

      {/* Floating particles */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-maroon rounded-full animate-float"></div>
        <div className="absolute top-1/3 right-1/4 w-1 h-1 bg-maroon-light rounded-full animate-float" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-accent rounded-full animate-float" style={{ animationDelay: '2s' }}></div>
        <div className="absolute top-2/3 right-1/3 w-1 h-1 bg-maroon-glow rounded-full animate-float" style={{ animationDelay: '0.5s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-1 h-1 bg-navy-light rounded-full animate-float" style={{ animationDelay: '1.5s' }}></div>
      </div>

      <div className="container mx-auto px-6 py-12 relative z-10">
        {/* Hero Section */}
        <div className="text-center mb-16 animate-slide-up">
          <div className="flex items-center justify-center gap-4 mb-6">
            <div className="w-16 h-16 bg-gradient-to-br from-maroon to-maroon-light rounded-2xl flex items-center justify-center animate-pulse-glow">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-5xl lg:text-7xl font-black">
              <span className="bg-gradient-to-r from-red-600 to-red-400 bg-clip-text text-transparent">STRYKER</span>
            </h1>
            <div className="w-16 h-16 bg-gradient-to-br from-navy to-navy-light rounded-2xl flex items-center justify-center animate-pulse-glow" style={{ animationDelay: '0.5s' }}>
              <Zap className="w-8 h-8 text-white" />
            </div>
          </div>

          <p className="text-2xl lg:text-3xl text-muted-foreground mb-8 max-w-4xl mx-auto leading-relaxed">
            Experience the future of soccer analysis with
            <span className="text-gradient-primary font-bold"> cutting-edge AI technology</span>
          </p>

          {/* Key features */}
          <div className="flex flex-wrap justify-center gap-6 mb-12">
            <div className="flex items-center gap-2 glass-card px-6 py-3 rounded-full hover-glow">
              <Target className="w-5 h-5 text-maroon" />
              <span className="text-lg font-semibold">Real-time Analysis</span>
            </div>
            <div className="flex items-center gap-2 glass-card px-6 py-3 rounded-full hover-glow">
              <Sparkles className="w-5 h-5 text-maroon" />
              <span className="text-lg font-semibold">Player Tracking</span>
            </div>
            <div className="flex items-center gap-2 glass-card px-6 py-3 rounded-full hover-glow">
              <Brain className="w-5 h-5 text-maroon" />
              <span className="text-lg font-semibold">Tactical Insights</span>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="space-y-24">
          {/* Upload Section */}
          <section className="animate-slide-up" style={{ animationDelay: '0.2s' }}>
            <div className="text-center mb-12">
              <h2 className="text-3xl lg:text-4xl font-black mb-4">
                <span className="text-gradient-primary">Upload & Analyze</span>
              </h2>
              <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                Drop your soccer footage and watch AI transform it into actionable insights
              </p>
            </div>

            <VideoUpload
              onVideoSelect={handleVideoUpload}
              isProcessing={isProcessing}
              progress={uploadProgress}
            />
          </section>

          {/* Gallery Section */}
          <section className="animate-slide-up" style={{ animationDelay: '0.4s' }}>
            <VideoGallery onVideoSelect={handleSampleVideoSelect} />
          </section>

        </div>
      </div>

      {/* Bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent"></div>
    </div>
  );
};

export default Index;
