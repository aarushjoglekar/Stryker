import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Maximize, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';

interface AnalysisData {
  events: Array<{
    action: string;
    confidence: number;
  }>;
}

interface VideoAnalysisProps {
  videoPath?: string;
  matchTitle?: string;
  matchDescription?: string;
  onClose: () => void;
}

const VideoAnalysis: React.FC<VideoAnalysisProps> = ({
  videoPath,
  matchTitle,
  matchDescription,
  onClose,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load video metadata and sync time updates
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoaded = () => {
      setDuration(video.duration);
      setIsLoading(false);
    };
    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('loadedmetadata', handleLoaded);
    video.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoaded);
      video.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [videoPath]);

  // Mock analysis data (replace with real API call)
  useEffect(() => {
    if (!videoPath) return;
    setIsLoading(true);

    // Call the Flask server at localhost:5000
    fetch('http://localhost:5000/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoPath }),
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const data: AnalysisData = await res.json();
        setAnalysisData(data);
      })
      .catch((err) => console.error('Prediction error:', err))
      .finally(() => setIsLoading(false));
  }, [videoPath]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (isPlaying) {
      video.pause();
      setIsPlaying(false);
    } else {
      video.play();
      setIsPlaying(true);
    }
  };

  const handleRestart = () => {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    if (!isPlaying) {
      video.play();
      setIsPlaying(true);
    }
  };

  // const handleFullscreen = () => {
  //   const container = containerRef.current;
  //   if (!container) return;
  //   if (document.fullscreenElement) {
  //     document.exitFullscreen();
  //   } else {
  //     container.requestFullscreen();
  //   }
  // };

  // const formatTime = (seconds: number) => {
  //   const mins = Math.floor(seconds / 60);
  //   const secs = Math.floor(seconds % 60);
  //   return `${mins}:${secs.toString().padStart(2, '0')}`;
  // };

  return (
    <div ref={containerRef} className="fixed inset-0 bg-background/95 backdrop-blur-sm z-50 overflow-auto">
      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div className="flex flex-col space-y-1">
            <h1 className="text-3xl font-black text-white">{matchTitle}</h1>
            <p className="text-sm text-muted-foreground">{matchDescription}</p>
          </div>
          <Button variant="ghost" onClick={onClose}>✕</Button>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Video Player */}
          <div className="lg:col-span-2">
            <div className="glass-card rounded-2xl overflow-hidden">
              <div className="relative aspect-video bg-black">
                {/* Video element */}
                {videoPath && (
                  <video
                    ref={videoRef}
                    src={videoPath}
                    className="w-full h-full object-contain"
                    controls={true}
                  />
                )}
                {/* Controls overlay */}
                {/* <div className="absolute bottom-4 left-4 right-4">
                  <div className="glass-card p-4 rounded-lg">
                    <div className="flex items-center gap-4 mb-3">
                      <Button variant="ghost" size="sm" onClick={togglePlay}>
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </Button>
                      <Button variant="ghost" size="sm" onClick={handleRestart} title="Restart">
                        <RotateCcw className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={handleFullscreen}>
                        <Maximize className="w-4 h-4" />
                      </Button>
                      <div className="flex-1">
                        <Progress
                          value={(currentTime / (duration || 1)) * 100}
                          className="h-1"
                        />
                      </div>
                      <span className="text-sm text-foreground">
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </span>
                    </div>
                  </div>
                </div> */}
              </div>
            </div>
          </div>

          {/* Events Panel */}
          <div className="space-y-6 lg:col-span-1">
            {isLoading ? (
              <div className="glass-card p-6 rounded-2xl text-center">
                <div className="w-16 h-16 bg-maroon/20 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse-glow">
                  <Activity className="w-8 h-8 text-maroon animate-spin" />
                </div>
                <h3 className="font-bold text-lg mb-2">Processing Video...</h3>
                <p className="text-muted-foreground text-sm">
                  AI is extracting key events from the match
                </p>
                <Progress value={75} className="mt-4" />
              </div>
            ) : (
              <div className="glass-card p-6 rounded-2xl">
                <h3 className="font-bold mb-4 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-maroon" />
                  Key Events
                </h3>
                <div className="space-y-3">
                  {analysisData?.events.map((event, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 glass rounded-lg">
                      <div className="text-sm font-mono text-maroon">
                        {index + 1}
                      </div>
                      <div className="flex-1">
                        <div className="text-sm font-medium">{event.action}</div>
                        <div className="text-xs text-muted-foreground">
                          Confidence: {Math.round(event.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoAnalysis;