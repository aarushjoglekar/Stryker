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

function round(float:number){
  return Math.round(float * 100) / 100;
}

const VideoAnalysis: React.FC<VideoAnalysisProps> = ({
  videoPath,
  matchTitle,
  matchDescription,
  onClose,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'events' | 'process'>('events');
  const [process, setProcess] = useState("")

  useEffect(() => {
    if (!videoPath) return;

    const analyze = async () => {
      setIsLoading(true);
      try {
        const fileResp = await fetch(videoPath);
        if (!fileResp.ok) throw new Error(`Couldn’t fetch video: ${fileResp.status}`);
        const blob = await fileResp.blob();
        const file = new File(
          [blob],
          videoPath.split('/').pop() || 'upload.mp4',
          { type: blob.type }
        );

        const formData = new FormData();
        formData.append('video', file);

        const res = await fetch('http://localhost:5001/api/predict', {
          method: 'POST',
          body: formData,
        });
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({ error: 'Invalid JSON' }));
          console.error('Flask error payload:', errBody);
          throw new Error(`Server error ${res.status}`);
        }
        const data: AnalysisData = await res.json();
        setAnalysisData(data);

        let process = "I'm clipping the first 60 seconds of the selected soccer video...\n\n" +
          "I'm unpacking the soccer video at 1FPS into 60 individual frames...\n\n" +
          "I'm now going to extract visual features from the pretrained Vision Transformer: DINOv2...\n\n" +
          "I've finished extracting features in the dimension 60x384 -> 384 features for each of the 60 frames!\n\n" +
          "I'll now pass the extracted visual features through a 6-layer transformer...\n\n" +
          "I've extracted a 1x17 vector of outputs from the transformer:\n" +
          "[\n" +
          "\t" + round(data['output'][0]) + " ← 'score' of class 0\n" +
          "\t" + round(data['output'][1]) +" ← 'score' of class 1\n" +
          "\t" + round(data['output'][2]) +" ← 'score' of class 2\n" +
          "\t" + round(data['output'][3]) +" ← 'score' of class 3\n" +
          "\t" + round(data['output'][4]) +" ← 'score' of class 4\n" +
          "\t" + round(data['output'][5]) +" ← 'score' of class 5\n" +
          "\t" + round(data['output'][6]) +" ← 'score' of class 6\n" +
          "\t" + round(data['output'][7]) +" ← 'score' of class 7\n" +
          "\t" + round(data['output'][8]) +" ← 'score' of class 8\n" +
          "\t" + round(data['output'][9]) +" ← 'score' of class 9\n" +
          "\t" + round(data['output'][10]) +" ← 'score' of class 10\n" +
          "\t" + round(data['output'][11]) +" ← 'score' of class 11\n" +
          "\t" + round(data['output'][12]) +" ← 'score' of class 12\n" +
          "\t" + round(data['output'][13]) +" ← 'score' of class 13\n" +
          "\t" + round(data['output'][14]) +" ← 'score' of class 14\n" +
          "\t" + round(data['output'][15]) +" ← 'score' of class 15\n" +
          "\t" + round(data['sigmoid'][16]) +" ← 'score' of class 16\n" +
          "]\n\n" +
          "I'll treat each of these 17 outputs as a representative for each action. Putting these through the sigmoid function should give me the probability of each action occurring...\n\n" +
          "I've calculated probabilities for each event occurring:\n" +
          "[\n" +
          "\t" + round(data['sigmoid'][0]) + " ← probability of class 0\n" +
          "\t" + round(data['sigmoid'][1]) +" ← probability of class 1\n" +
          "\t" + round(data['sigmoid'][2]) +" ← probability of class 2\n" +
          "\t" + round(data['sigmoid'][3]) +" ← probability of class 3\n" +
          "\t" + round(data['sigmoid'][4]) +" ← probability of class 4\n" +
          "\t" + round(data['sigmoid'][5]) +" ← probability of class 5\n" +
          "\t" + round(data['sigmoid'][6]) +" ← probability of class 6\n" +
          "\t" + round(data['sigmoid'][7]) +" ← probability of class 7\n" +
          "\t" + round(data['sigmoid'][8]) +" ← probability of class 8\n" +
          "\t" + round(data['sigmoid'][9]) +" ← probability of class 9\n" +
          "\t" + round(data['sigmoid'][10]) +" ← probability of class 10\n" +
          "\t" + round(data['sigmoid'][11]) +" ← probability of class 11\n" +
          "\t" + round(data['sigmoid'][12]) +" ← probability of class 12\n" +
          "\t" + round(data['sigmoid'][13]) +" ← probability of class 13\n" +
          "\t" + round(data['sigmoid'][14]) +" ← probability of class 14\n" +
          "\t" + round(data['sigmoid'][15]) +" ← probability of class 15\n" +
          "\t" + round(data['sigmoid'][16]) +" ← probability of class 16\n" +
          "]\n\n" +
          "I'll classify probabilities that are greater than 0.5 have occurred in the clip....\n\n" +
          "I have made my predictions! Check the events tab and play the video to see if I was right!";

        setProcess(process)
      } catch (err) {
        console.error('Upload / prediction error:', err);
      } finally {
        setIsLoading(false);
      }
    };

    analyze();
  }, [videoPath]);

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
                {videoPath && (
                  <video
                    ref={videoRef}
                    src={videoPath}
                    className="w-full h-full object-contain"
                    controls
                  />
                )}
              </div>
            </div>
          </div>

          {/* Events & Process Tabs */}
          <div className="lg:col-span-1">
            <div className="glass-card p-6 rounded-2xl bg-black">
              {/* Tab Navigator */}
              <div className="flex w-full bg-black rounded-lg mb-4 overflow-hidden">
                <button
                  onClick={() => setActiveTab('events')}
                  className={`flex-1 py-2 text-center transition ${activeTab === 'events'
                    ? 'bg-maroon/50 text-white'
                    : 'text-muted-foreground hover:bg-gray-800/20'
                    }`}
                >
                  Events
                </button>
                <button
                  onClick={() => setActiveTab('process')}
                  className={`flex-1 py-2 text-center transition ${activeTab === 'process'
                    ? 'bg-maroon/50 text-white'
                    : 'text-muted-foreground hover:bg-gray-800/20'
                    }`}
                >
                  Process
                </button>
              </div>

              {/* Tab Content */}
              {isLoading ? (
                <div className="text-center">
                  <div className="w-16 h-16 bg-maroon/20 rounded-full flex items-center justify-center mx-auto mb-4 animate-pulse-glow">
                    <Activity className="w-8 h-8 text-maroon animate-spin" />
                  </div>
                  <h3 className="font-bold text-lg mb-2 text-white">Processing Video...</h3>
                  <p className="text-muted-foreground text-sm">
                    AI is extracting key events from the match
                  </p>
                  <Progress value={75} className="mt-4" />
                </div>
              ) : activeTab === 'events' ? (
                <div className="space-y-3">
                  {analysisData?.events.map((event, index) => (
                    <div key={index} className="flex items-center gap-3 p-3 glass rounded-lg">
                      <div className="text-sm font-mono text-maroon">{index + 1}</div>
                      <div className="flex-1">
                        <div className="text-sm font-medium">{event.action}</div>
                        <div className="text-xs text-muted-foreground">
                          Confidence: {Math.round(event.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                  ))}
                  {!analysisData && (
                    <p className="text-sm text-muted-foreground">No events to display.</p>
                  )}
                </div>
              ) : (
                <div className="text-center">
                  <pre className="text-sm text-white whitespace-pre-wrap text-left">{process}</pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoAnalysis;
