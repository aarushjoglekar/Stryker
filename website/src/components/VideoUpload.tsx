import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Play, Loader2, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';

interface VideoUploadProps {
  onVideoSelect: (file: File) => void;
  isProcessing: boolean;
  progress: number;
}

const VideoUpload: React.FC<VideoUploadProps> = ({ onVideoSelect, isProcessing, progress }) => {
  const { toast } = useToast();
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0];
      if (file) {
        // Create preview URL
        const url = URL.createObjectURL(file);
        setPreviewUrl(url);
        
        // Trigger analysis
        onVideoSelect(file);
        
        toast({
          title: "Video uploaded!",
          description: "Starting AI analysis...",
        });
      }
    },
    [onVideoSelect, toast]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.webm']
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false
  });

  return (
    <div className="w-full">
      <div
        {...getRootProps()}
        className={`
          relative group cursor-pointer transition-all duration-300
          glass-card p-12 rounded-3xl border-2 border-dashed
          ${isDragActive 
            ? 'border-maroon bg-maroon/10 scale-105' 
            : 'border-maroon/30 hover:border-maroon/60 hover:bg-maroon/5'
          }
          ${isProcessing ? 'pointer-events-none opacity-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {/* Background glow effect */}
        <div className="absolute inset-0 bg-gradient-to-br from-maroon/20 via-transparent to-navy/20 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>
        
        <div className="relative z-10 flex flex-col items-center justify-center space-y-6">
          {previewUrl && !isProcessing ? (
            <div className="w-full max-w-md">
              <video
                src={previewUrl}
                className="w-full h-48 object-cover rounded-2xl border border-maroon/30"
                controls
                muted
              />
            </div>
          ) : (
            <div className="relative">
              <div className={`
                w-24 h-24 rounded-full glass-card flex items-center justify-center mb-4
                ${isProcessing ? 'animate-pulse-glow' : 'group-hover:scale-110'}
                transition-transform duration-300
              `}>
                {isProcessing ? (
                  <Loader2 className="w-12 h-12 text-maroon animate-spin" />
                ) : isDragActive ? (
                  <Zap className="w-12 h-12 text-maroon animate-bounce" />
                ) : (
                  <Upload className="w-12 h-12 text-maroon group-hover:text-maroon-light transition-colors" />
                )}
              </div>
            </div>
          )}

          <div className="text-center space-y-2">
            {isProcessing ? (
              <>
                <h3 className="text-2xl font-bold text-gradient-primary">Processing Video...</h3>
                <p className="text-muted-foreground">AI is analyzing your soccer footage</p>
                <div className="w-full max-w-xs mx-auto mt-4">
                  <Progress value={progress} className="h-2" />
                  <p className="text-sm text-muted-foreground mt-2">{progress}% complete</p>
                </div>
              </>
            ) : isDragActive ? (
              <>
                <h3 className="text-2xl font-bold text-gradient-primary">Drop your video here!</h3>
                <p className="text-muted-foreground">Release to start AI analysis</p>
              </>
            ) : (
              <>
                <h3 className="text-2xl font-bold text-gradient-primary">Upload Soccer Video</h3>
                <p className="text-muted-foreground">
                  Drag & drop your match footage or training session
                </p>
                <p className="text-sm text-muted-foreground">
                  Supports MP4, AVI, MOV, WebM (max 100MB)
                </p>
              </>
            )}
          </div>

          {!isProcessing && (
            <Button variant="hero" size="lg" className="mt-4">
              <Play className="w-5 h-5 mr-2" />
              Choose Video File
            </Button>
          )}
        </div>

        {/* Animated particles */}
        {!isProcessing && (
          <div className="absolute inset-0 overflow-hidden rounded-3xl pointer-events-none">
            <div className="absolute top-1/4 left-1/4 w-2 h-2 bg-maroon rounded-full opacity-60 animate-float"></div>
            <div className="absolute top-1/3 right-1/4 w-1 h-1 bg-maroon-light rounded-full opacity-40 animate-float" style={{animationDelay: '1s'}}></div>
            <div className="absolute bottom-1/4 left-1/3 w-1.5 h-1.5 bg-accent rounded-full opacity-50 animate-float" style={{animationDelay: '2s'}}></div>
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoUpload;