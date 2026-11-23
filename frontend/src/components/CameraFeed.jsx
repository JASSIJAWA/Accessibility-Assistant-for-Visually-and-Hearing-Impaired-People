import { useRef, useEffect, useState, useCallback } from 'react';

const getWebSocketUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/vision`;
};

const BACKEND_WS = getWebSocketUrl();

function CameraFeed({ onDetectionUpdate, isActive = true }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const intervalRef = useRef(null);
  const isActiveRef = useRef(isActive);

  useEffect(() => {
    isActiveRef.current = isActive;
    if (!isActive) {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      if (intervalRef.current) clearTimeout(intervalRef.current);
    }
  }, [isActive]);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [detections, setDetections] = useState({ objects: [], emotions: [] });

  // Fullscreen & Guidance State
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [spokenSubtitles, setSpokenSubtitles] = useState('');
  const [narrationEnabled, setNarrationEnabled] = useState(false);

  // OCR State
  const [isReadingText, setIsReadingText] = useState(false);
  const [recognizedText, setRecognizedText] = useState(null);

  // OCR Debug State
  const [ocrDebug, setOcrDebug] = useState({
    imageUrl: null,
    rawText: null,
    confidence: null,
    filteredText: null
  });

  const lastGeminiRef = useRef('');
  const narrationTimerRef = useRef(null);

  const [facingMode, setFacingMode] = useState('environment');
  const [aspectRatio, setAspectRatio] = useState(4/3);

  // Start sending frames once streaming
  useEffect(() => {
    let isRunning = true;

    const captureAndSend = async () => {
      if (!isRunning) return;

      // If ready and vision tab is active, send a frame
      if (isActiveRef.current && isStreaming && isConnected && videoRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (canvas && video.videoWidth > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0);
          const base64 = canvas.toDataURL('image/jpeg', 0.5);
          wsRef.current.send(base64);
        }
      }

      intervalRef.current = setTimeout(captureAndSend, 1000);
    };

    if (isStreaming && isConnected) {
      captureAndSend();
    }

    return () => {
      isRunning = false;
      if (intervalRef.current) clearTimeout(intervalRef.current);
    };
  }, [isStreaming, isConnected]);

  // Connect to backend WebSocket
  const connectWebSocket = useCallback(() => {
    try {
      const ws = new WebSocket(BACKEND_WS);

      ws.onopen = () => {
        console.log('✅ Connected to Vision AI backend');
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const result = JSON.parse(event.data);
          if (!result.error) {
            setDetections(result);
            if (onDetectionUpdate) onDetectionUpdate(result);
          }

          // FAST LOOP: As soon as we get a result, instantly send the NEXT frame
          if (isActiveRef.current && videoRef.current && videoRef.current.readyState >= 2 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            if (canvas && video.videoWidth > 0) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
              const ctx = canvas.getContext('2d');
              ctx.drawImage(video, 0, 0);
              const base64 = canvas.toDataURL('image/jpeg', 0.5);
              wsRef.current.send(base64);

              // Reset fallback timeout
              if (intervalRef.current) clearTimeout(intervalRef.current);
              intervalRef.current = setTimeout(() => {
                if (isActiveRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                  wsRef.current.send(base64);
                }
              }, 1000);
            }
          }
        } catch (e) {
          console.error('Failed to parse AI result:', e);
        }
      };

      ws.onclose = () => {
        console.log('❌ Vision WebSocket disconnected');
        setIsConnected(false);
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = () => {
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('WebSocket connection failed:', e);
      setIsConnected(false);
    }
  }, [onDetectionUpdate]);

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
  };

  // Start webcam
  const startCamera = async (mode = facingMode) => {
    stopCamera();
    setError(null);

    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const videoConstraints = {
      facingMode: { ideal: mode },
      width: isMobile ? { ideal: 1280 } : { ideal: 1920 },
      height: isMobile ? { ideal: 720 } : { ideal: 1080 }
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play().then(() => {
            setIsStreaming(true);
            setAspectRatio(videoRef.current.videoWidth / videoRef.current.videoHeight);
          }).catch(e => {
            console.error("Play error:", e);
            setError("Could not autoplay camera. Please tap to enable.");
          });
        };
      }
    } catch (err) {
      console.warn("High-res camera failed, falling back to basic:", err);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: mode },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play().then(() => {
              setIsStreaming(true);
              setAspectRatio(videoRef.current.videoWidth / videoRef.current.videoHeight);
            });
          };
        }
      } catch (fallbackErr) {
        console.error("Webcam access error:", fallbackErr);
        setError("Camera access denied or unavailable. Please check permissions.");
        setIsStreaming(false);
      }
    }
  };

  const toggleCamera = () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(nextMode);
    startCamera(nextMode);
  };

  // ─── Guidance Toggle (Starts Fullscreen + Voice Guidance) ──────────────────
  const toggleGuidance = () => {
    if (narrationEnabled) {
      setNarrationEnabled(false);
      setIsFullscreen(false);
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    } else {
      setNarrationEnabled(true);
      setIsFullscreen(true);
      const startMsg = "Guidance started. Point camera forward.";
      setSpokenSubtitles(startMsg);
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(startMsg);
        utterance.rate = 1.1;
        window.speechSynthesis.speak(utterance);
      }
    }
  };

  // Keyboard Escape listener
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false);
        setNarrationEnabled(false);
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFullscreen]);

  // Capture frame and call OCR endpoint
  const handleReadText = async () => {
    if (!videoRef.current || !canvasRef.current || isReadingText) return;

    setIsReadingText(true);
    setRecognizedText(null);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      const imageUrl = canvas.toDataURL('image/jpeg', 0.95);

      const res = await fetch('/api/ocr', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageUrl }),
      });

      const data = await res.json();

      if (data.error) {
        setRecognizedText(`Error: ${data.error}`);
        return;
      }

      const isAccepted = data.accepted && data.text && data.text.trim().length > 0;
      setRecognizedText(isAccepted ? data.text : "No clear text detected. Hold camera steady.");

      const wordDetails = (data.results || []).map(r => ({
        text: r.text,
        confidence: r.confidence,
        accepted: r.accepted
      }));

      setOcrDebug({
        imageUrl,
        rawText: (data.results || []).map(r => r.text).join(' '),
        confidence: wordDetails.length > 0 ? Math.round(wordDetails.reduce((sum, w) => sum + w.confidence, 0) / wordDetails.length) : 0,
        filteredText: isAccepted ? data.text : '',
        status: isAccepted ? 'ACCEPTED' : 'REJECTED',
        wordDetails
      });

      if (isAccepted) {
        let speakableText = data.text;
        if (speakableText.length > 200) {
          speakableText = speakableText.substring(0, 200) + "... and more.";
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(speakableText);
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
      }

    } catch (err) {
      console.error('OCR Error:', err);
      setRecognizedText("OCR Failed. Please try again.");
    } finally {
      setIsReadingText(false);
    }
  };

  // Initialize camera and WebSocket on mount
  useEffect(() => {
    startCamera(facingMode);
    connectWebSocket();

    return () => {
      stopCamera();
      if (wsRef.current) wsRef.current.close();
      if (intervalRef.current) clearTimeout(intervalRef.current);
    };
  }, []);

  // Gemini Scene Description — rich, natural guidance
  useEffect(() => {
    if (!isActive || !narrationEnabled || isReadingText) {
      if (!isActive && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      return;
    }
    const desc = detections.scene_description;
    if (!desc || desc === lastGeminiRef.current) return;
    lastGeminiRef.current = desc;
    setSpokenSubtitles(desc);

    // Gemini descriptions get priority
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(desc);
    utterance.rate = 1.05;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);
  }, [detections.scene_description, narrationEnabled, isReadingText, isActive]);

  // YOLO Fallback Narrator — short alerts when Gemini hasn't spoken yet
  useEffect(() => {
    if (!isActive || !narrationEnabled || isReadingText) return;
    if (!detections.narration || !detections.scene_changed) return;
    if (detections.scene_description) return;

    const narration = detections.narration;

    if (narrationTimerRef.current) return;
    narrationTimerRef.current = setTimeout(() => {
      narrationTimerRef.current = null;
    }, 4000);

    setSpokenSubtitles(narration);
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(narration);
    utterance.rate = 1.15;
    utterance.volume = 0.9;
    window.speechSynthesis.speak(utterance);
  }, [detections, narrationEnabled, isReadingText, isActive]);

  // Shared Video & Overlays Component
  const renderVideoAndOverlays = (isFs) => (
    <div className={`relative flex-1 w-full h-full min-h-0 flex items-center justify-center overflow-hidden ${isFs ? 'bg-black' : 'bg-[#050510]'}`}>
      {error ? (
        <div className="absolute inset-0 flex items-center justify-center text-center text-red-400 p-8 z-10">
          <div>
            <p className="font-semibold text-sm">⚠️ {error}</p>
            <button
              onClick={() => startCamera(facingMode)}
              className="mt-4 px-6 py-2 bg-purple-500/30 border border-purple-500/40 rounded-xl text-white text-sm hover:bg-purple-500/50 transition-colors"
            >
              Retry Camera
            </button>
          </div>
        </div>
      ) : (
        <div
          className="relative flex-shrink-0"
          style={{
            aspectRatio: aspectRatio,
            maxHeight: '100%',
            maxWidth: '100%'
          }}
        >
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={(e) => {
              if (e.target.videoWidth && e.target.videoHeight) {
                setAspectRatio(e.target.videoWidth / e.target.videoHeight);
              }
            }}
            className={`w-full h-full object-contain ${facingMode === 'user' ? '-scale-x-100' : ''}`}
          />

          {/* AI Bounding Box Overlays */}
          <div className={`absolute inset-0 pointer-events-none ${facingMode === 'user' ? '-scale-x-100' : ''}`}>
            {/* Object boxes — color coded by urgency */}
            {detections.objects?.map((obj, i) => {
              const fs = detections.frame_size;
              if (!fs) return null;
              const style = {
                left: `${(obj.box.x1 / fs.width) * 100}%`,
                top: `${(obj.box.y1 / fs.height) * 100}%`,
                width: `${((obj.box.x2 - obj.box.x1) / fs.width) * 100}%`,
                height: `${((obj.box.y2 - obj.box.y1) / fs.height) * 100}%`,
              };
              const urgencyColors = {
                critical: 'border-red-500 bg-red-600/90 shadow-[0_0_12px_rgba(239,68,68,0.6)]',
                warning:  'border-amber-400 bg-amber-500/90 shadow-[0_0_10px_rgba(245,158,11,0.5)]',
                info:     'border-emerald-400 bg-emerald-500/80',
                background: 'border-gray-400 bg-gray-500/70',
              };
              const colors = urgencyColors[obj.urgency] || urgencyColors.info;
              const [borderColor, bgColor] = colors.split(' ');
              const distText = obj.distance_m ? `${obj.distance_m}m` : `~${obj.distance_steps}steps`;
              return (
                <div key={`obj-${i}`} className={`absolute border-2 ${borderColor} rounded-sm transition-all duration-300 ease-linear`} style={style}>
                  <span className={`absolute -top-5 left-0 ${bgColor} text-white font-bold text-[0.62rem] md:text-[0.68rem] px-1.5 py-0.5 rounded whitespace-nowrap ${facingMode === 'user' ? '-scale-x-100' : ''}`}>
                    {obj.urgency === 'critical' ? '⚠️ ' : ''}{obj.label} • {distText} {obj.position ? `(${obj.position})` : ''}
                  </span>
                </div>
              );
            })}

            {/* Emotion boxes */}
            {detections.emotions?.map((face, i) => {
              const fs = detections.frame_size;
              if (!fs) return null;
              const style = {
                left: `${(face.box.x / fs.width) * 100}%`,
                top: `${(face.box.y / fs.height) * 100}%`,
                width: `${(face.box.w / fs.width) * 100}%`,
                height: `${(face.box.h / fs.height) * 100}%`,
              };
              const emotionColors = {
                happy: 'border-yellow-400 bg-yellow-500/80',
                sad: 'border-blue-400 bg-blue-500/80',
                angry: 'border-red-400 bg-red-500/80',
                surprise: 'border-pink-400 bg-pink-500/80',
                fear: 'border-orange-400 bg-orange-500/80',
                disgust: 'border-green-400 bg-green-500/80',
                neutral: 'border-gray-400 bg-gray-500/80',
              };
              const colors = emotionColors[face.emotion] || 'border-purple-400 bg-purple-500/80';
              const [borderColor, bgColor] = colors.split(' ');

              return (
                <div key={`emo-${i}`} className={`absolute border-2 ${borderColor} rounded-sm transition-all duration-300 ease-linear`} style={style}>
                  <span className={`absolute -top-5 left-0 ${bgColor} text-white font-bold text-[0.62rem] px-1.5 py-0.5 rounded whitespace-nowrap ${facingMode === 'user' ? '-scale-x-100' : ''}`}>
                    {face.emotion} {Math.round(face.confidence)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <>
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* ═══════════════════════════════════════════════════════════════════════
          FULLSCREEN GUIDANCE MODE (Active when isFullscreen is true)
          ═══════════════════════════════════════════════════════════════════════ */}
      {isFullscreen ? (
        <div className="fixed inset-0 z-50 bg-black flex flex-col w-full h-[100dvh] overflow-hidden select-none">

          {/* Top HUD Floating Control Bar */}
          <div className="absolute top-0 left-0 right-0 z-20 flex items-center justify-between p-3 md:p-4 bg-gradient-to-b from-black/90 via-black/60 to-transparent pointer-events-auto">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping" />
              <span className="text-xs md:text-sm font-bold text-emerald-300 bg-emerald-950/80 border border-emerald-500/40 px-3 py-1 rounded-full shadow-lg">
                ● GEMINI LIVE GUIDANCE
              </span>
              <span className="text-[0.68rem] text-white/50 bg-black/40 px-2 py-0.5 rounded border border-white/10 hidden sm:inline">
                {facingMode === 'environment' ? 'Rear Cam' : 'Front Cam'}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={toggleCamera}
                className="px-3 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 border border-white/20 text-white text-xs font-semibold flex items-center gap-1.5 backdrop-blur-md transition-all cursor-pointer"
                title="Switch Camera"
              >
                <span>🔄</span>
                <span className="hidden sm:inline">Flip</span>
              </button>

              <button
                onClick={handleReadText}
                disabled={isReadingText}
                className="px-3 py-1.5 rounded-xl bg-blue-600/90 hover:bg-blue-600 border border-blue-400/50 text-white text-xs font-semibold flex items-center gap-1.5 backdrop-blur-md transition-all shadow-lg shadow-blue-500/20 cursor-pointer"
                title="Read Text via OCR"
              >
                <span>📖</span>
                <span>{isReadingText ? 'Reading...' : 'Read Text'}</span>
              </button>

              <button
                onClick={toggleGuidance}
                className="px-3.5 py-1.5 rounded-xl bg-red-600 hover:bg-red-500 border border-red-400 text-white text-xs font-bold flex items-center gap-1.5 shadow-[0_0_15px_rgba(239,68,68,0.5)] transition-all cursor-pointer"
                title="Stop Guidance and Exit Fullscreen"
              >
                <span>✕</span>
                <span>GUIDE OFF</span>
              </button>
            </div>
          </div>

          {/* Center-Top Gemini Spoken Guidance HUD Banner */}
          {(detections.scene_description || spokenSubtitles) && (
            <div className="absolute top-16 md:top-20 left-3 right-3 md:left-1/2 md:-translate-x-1/2 md:max-w-2xl z-20 pointer-events-none">
              <div className="bg-black/85 backdrop-blur-xl border border-purple-500/50 rounded-2xl p-3 md:p-4 shadow-[0_0_30px_rgba(168,85,247,0.4)] flex items-start gap-3">
                <span className="text-2xl flex-shrink-0">🧠</span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-[0.65rem] font-bold uppercase tracking-wider text-purple-300">
                      Gemini Real-Time Guide:
                    </span>
                    <span className="text-[0.62rem] text-emerald-400 font-mono flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                      Speaking
                    </span>
                  </div>
                  <p className="text-sm md:text-base font-semibold text-gray-100 leading-relaxed tracking-wide">
                    {detections.scene_description || spokenSubtitles}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Fullscreen Video Area */}
          {renderVideoAndOverlays(true)}

          {/* OCR Result Floating Modal (in Fullscreen) */}
          {recognizedText && (
            <div className="absolute inset-x-4 top-24 z-30 bg-blue-950/90 backdrop-blur-xl border border-blue-400/50 rounded-2xl p-4 shadow-2xl">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-bold text-blue-300 uppercase tracking-wider flex items-center gap-1.5">
                  <span>📖</span> Detected Text (OCR)
                </span>
                <button
                  onClick={() => setRecognizedText(null)}
                  className="text-white/60 hover:text-white text-xs font-bold px-2 py-1 bg-white/10 rounded-lg cursor-pointer"
                >
                  ✕ Close
                </button>
              </div>
              <p className="text-sm md:text-base text-gray-100 whitespace-pre-wrap font-medium leading-relaxed">
                {recognizedText}
              </p>
            </div>
          )}

          {/* Bottom Floating Obstacle Radar Bar */}
          <div className="absolute bottom-0 left-0 right-0 z-20 p-3 md:p-4 bg-gradient-to-t from-black/95 via-black/70 to-transparent flex flex-col gap-2 pointer-events-auto">
            {/* Detected Obstacles Chips */}
            {detections.objects && detections.objects.length > 0 && (
              <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
                <span className="text-[0.65rem] text-white/50 font-bold uppercase flex-shrink-0 mr-1">
                  Radar:
                </span>
                {detections.objects.slice(0, 6).map((obj, i) => {
                  const isCritical = obj.urgency === 'critical';
                  const chipColor = isCritical
                    ? 'bg-red-500/30 border-red-500 text-red-200 animate-bounce'
                    : obj.urgency === 'warning'
                      ? 'bg-amber-500/20 border-amber-500/40 text-amber-200'
                      : 'bg-white/10 border-white/20 text-white/80';
                  return (
                    <span
                      key={i}
                      className={`text-xs px-2.5 py-1 rounded-lg border font-semibold whitespace-nowrap flex items-center gap-1 backdrop-blur-md ${chipColor}`}
                    >
                      {isCritical && <span>⚠️</span>}
                      <span>{obj.label}</span>
                      <span className="font-mono text-[0.7rem] opacity-80 font-bold">
                        {obj.distance_m ? `${obj.distance_m}m` : `~${obj.distance_steps}st`}
                      </span>
                      <span className="text-[0.65rem] opacity-60">({obj.position || 'ahead'})</span>
                    </span>
                  );
                })}
              </div>
            )}

            {/* Bottom Status & Controls */}
            <div className="flex items-center justify-between gap-3 pt-1 border-t border-white/10">
              <div className="flex items-center gap-2 text-xs text-white/70 min-w-0">
                <span className="text-base flex-shrink-0">🔊</span>
                <span className="truncate italic text-[0.75rem]">
                  {spokenSubtitles || 'Point phone forward — AI is observing path...'}
                </span>
              </div>

              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={() => {
                    setNarrationEnabled(!narrationEnabled);
                    if (narrationEnabled) window.speechSynthesis.cancel();
                  }}
                  className="px-2.5 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-medium border border-white/15 cursor-pointer"
                >
                  {narrationEnabled ? '🔊 Mute' : '🔇 Unmute'}
                </button>
                <button
                  onClick={() => setIsFullscreen(false)}
                  className="px-2.5 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-medium border border-white/15 cursor-pointer"
                >
                  ⛶ Minimize
                </button>
              </div>
            </div>
          </div>

        </div>
      ) : (

        /* ═════════════════════════════════════════════════════════════════════
           NORMAL DASHBOARD CARD MODE (When isFullscreen is false)
           ═════════════════════════════════════════════════════════════════════ */
        <div className="flex flex-col h-full overflow-hidden">

          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-white/5 flex-shrink-0">
            <div className="flex items-center gap-2 flex-1">
              <span className="text-lg">👁️</span>
              <h2 className="text-sm font-semibold text-gray-200">Vision Assistant</h2>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={toggleCamera}
                className="p-1.5 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-white text-xs flex items-center gap-1 px-2.5 cursor-pointer"
                title="Switch Camera"
              >
                <span>🔄</span>
                <span className="hidden sm:inline">{facingMode === 'environment' ? 'Rear' : 'Front'}</span>
              </button>

              <button
                onClick={() => setIsFullscreen(true)}
                className="p-1.5 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition-colors text-white text-xs flex items-center gap-1 px-2.5 cursor-pointer"
                title="Expand Camera to Fullscreen"
              >
                <span>⛶</span>
                <span className="hidden sm:inline">Expand</span>
              </button>

              <div className="flex items-center gap-1.5 ml-1">
                <span className={`text-[0.6rem] px-2 py-0.5 rounded-full font-medium ${
                  isConnected
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    : 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                }`}>
                  {isConnected ? 'AI Online' : 'AI Offline'}
                </span>
                <span className={`w-2.5 h-2.5 rounded-full ${isStreaming ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]' : 'bg-gray-600'}`} />
              </div>
            </div>
          </div>

          {/* Video in Card */}
          {renderVideoAndOverlays(false)}

          {/* Gemini Scene Description Bar */}
          {detections.scene_description && (
            <div className="px-4 py-2 border-t border-indigo-500/20 bg-indigo-500/5 flex-shrink-0">
              <p className="text-xs text-indigo-200 leading-relaxed">
                <span className="text-indigo-400 font-semibold">🧠 AI Guide:</span> {detections.scene_description}
              </p>
            </div>
          )}

          {/* Detected Objects Chips in Card */}
          {detections.objects && detections.objects.length > 0 && (
            <div className="px-3 py-1.5 border-t border-white/5 flex items-center gap-1.5 overflow-x-auto flex-shrink-0 bg-black/20">
              {detections.objects.slice(0, 5).map((obj, i) => {
                const tagColors = {
                  critical: 'bg-red-500/15 text-red-300 border-red-500/30',
                  warning:  'bg-amber-500/15 text-amber-300 border-amber-500/30',
                  info:     'bg-emerald-500/10 text-emerald-300 border-emerald-500/20',
                  background: 'bg-gray-500/10 text-gray-400 border-gray-500/20',
                };
                const tc = tagColors[obj.urgency] || tagColors.info;
                return (
                  <span key={`tag-${i}`} className={`text-[0.65rem] px-2 py-0.5 ${tc} rounded-md whitespace-nowrap border`}>
                    {obj.urgency === 'critical' ? '⚠️ ' : ''}{obj.label} {obj.distance_m ? `${obj.distance_m}m` : ''} {obj.position ? `(${obj.position})` : ''}
                  </span>
                );
              })}
            </div>
          )}

          {/* Main Action Bar */}
          <div className="px-4 py-3 border-t border-white/10 bg-[#0c0c22] flex-shrink-0 flex items-center gap-2.5">
            <button
              onClick={toggleGuidance}
              className="flex-1 py-3 px-4 rounded-xl bg-gradient-to-r from-amber-600 via-orange-600 to-amber-700 hover:from-amber-500 hover:to-orange-500 border border-amber-400 text-white font-bold text-xs sm:text-sm shadow-[0_0_20px_rgba(245,158,11,0.35)] flex items-center justify-center gap-2 transition-all cursor-pointer active:scale-95"
            >
              <span className="text-base sm:text-lg">🔊</span>
              <span>GUIDE ON (FULLSCREEN AI)</span>
            </button>

            <button
              onClick={handleReadText}
              disabled={isReadingText || !isStreaming}
              className="px-4 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 border border-blue-400 text-white font-bold text-xs sm:text-sm shadow-[0_0_15px_rgba(59,130,246,0.3)] flex items-center justify-center gap-1.5 transition-all cursor-pointer"
            >
              <span>📖</span>
              <span className="hidden sm:inline">{isReadingText ? 'READING...' : 'READ TEXT'}</span>
            </button>
          </div>

          {/* OCR Result in Card */}
          {recognizedText && (
            <div className="px-4 py-3 border-t border-white/5 bg-blue-500/10 flex-shrink-0">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-semibold text-blue-300 uppercase tracking-wider">Detected Text</span>
                <button onClick={() => setRecognizedText(null)} className="text-white/40 hover:text-white/80 text-xs">✕ Dismiss</button>
              </div>
              <p className="text-sm text-gray-200 whitespace-pre-wrap font-medium">
                {recognizedText}
              </p>
            </div>
          )}

        </div>
      )}
    </>
  );
}

export default CameraFeed;
