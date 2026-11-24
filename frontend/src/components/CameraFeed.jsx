import { useRef, useEffect, useState, useCallback } from 'react';

const getWebSocketUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/vision`;
};

const BACKEND_WS = getWebSocketUrl();

function CameraFeed({ onDetectionUpdate, isActive = true, isGuidanceActive = false, onToggleGuidance }) {
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

  // Guidance audio and subtitle state
  const [spokenSubtitles, setSpokenSubtitles] = useState('');
  const [isMuted, setIsMuted] = useState(false);

  // Flashlight / Torch State
  const [isTorchOn, setIsTorchOn] = useState(false);
  const [hasTorch, setHasTorch] = useState(false);
  const mediaStreamRef = useRef(null);

  // OCR State
  const [isReadingText, setIsReadingText] = useState(false);
  const [recognizedText, setRecognizedText] = useState(null);

  const lastGeminiRef = useRef('');
  const lastYoloAlertTimeRef = useRef(0);
  const audioCtxRef = useRef(null);
  const lastSonarPingRef = useRef(0);
  const mountedRef = useRef(true);  // Prevents zombie reconnects after unmount
  const [isSonarEnabled, setIsSonarEnabled] = useState(true);

  const [facingMode, setFacingMode] = useState('environment');
  const [aspectRatio, setAspectRatio] = useState(4/3);

  // Proximity Sonar Ping (Acoustic non-verbal radar for zero-delay spatial awareness)
  const playSonarPing = useCallback((freq = 840, duration = 0.04, volume = 0.06) => {
    if (isMuted || !isSonarEnabled) return;
    try {
      if (!audioCtxRef.current) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioCtxRef.current = new AudioContextClass();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        ctx.resume();
      }
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      gain.gain.setValueAtTime(volume, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
    } catch (e) {
      // AudioContext blocked until interaction or unsupported
    }
  }, [isMuted, isSonarEnabled]);

  // Safe TTS Speak Helper with Chrome Android resume support
  const safeSpeak = useCallback((text) => {
    if (isMuted || !text || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.15; // Snappy, prompt delivery
      utterance.volume = 1.0;
      if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
      }
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      console.warn("TTS Error:", e);
    }
  }, [isMuted]);

  // Downsample and send frame to backend
  const sendCurrentFrame = useCallback(() => {
    if (!isActiveRef.current || !videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!canvas || video.videoWidth === 0) return;

    // Scale to max 640px for ultra-low latency frame transmission (~25KB)
    const maxDim = 640;
    let targetW = video.videoWidth;
    let targetH = video.videoHeight;
    if (targetW > maxDim || targetH > maxDim) {
      if (targetW >= targetH) {
        targetH = Math.round((targetH * maxDim) / targetW);
        targetW = maxDim;
      } else {
        targetW = Math.round((targetW * maxDim) / targetH);
        targetH = maxDim;
      }
    }
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, targetW, targetH);
    const base64 = canvas.toDataURL('image/jpeg', 0.55);
    wsRef.current.send(base64);
  }, []);

  // Frame capture loop
  useEffect(() => {
    let isRunning = true;

    const captureAndSend = () => {
      if (!isRunning) return;
      if (isStreaming && isConnected) {
        sendCurrentFrame();
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
  }, [isStreaming, isConnected, sendCurrentFrame]);

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

          // FAST LOOP: Instantly send next frame once processed
          if (isActiveRef.current && videoRef.current && videoRef.current.readyState >= 2 && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            sendCurrentFrame();

            // Clear fallback timeout
            if (intervalRef.current) clearTimeout(intervalRef.current);
            intervalRef.current = setTimeout(() => {
              sendCurrentFrame();
            }, 1000);
          }
        } catch (e) {
          console.error('Failed to parse AI result:', e);
        }
      };

      ws.onclose = () => {
        console.log('❌ Vision WebSocket disconnected');
        setIsConnected(false);
        // Only reconnect if component is still mounted
        if (mountedRef.current) {
          setTimeout(connectWebSocket, 3000);
        }
      };

      ws.onerror = () => {
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (e) {
      console.error('WebSocket connection failed:', e);
      setIsConnected(false);
    }
  }, [onDetectionUpdate, sendCurrentFrame]);

  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
      setIsStreaming(false);
    }
    setIsTorchOn(false);
  };

  // Start webcam
  const startCamera = async (mode = facingMode) => {
    stopCamera();
    setError(null);

    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
    const isPortrait = typeof window !== 'undefined' && window.innerHeight > window.innerWidth;

    // In portrait orientation on mobile, request portrait resolution (e.g. 720x1280) so the camera doesn't letterbox into a tiny strip
    const videoConstraints = {
      facingMode: { ideal: mode },
      width: isMobile ? (isPortrait ? { ideal: 720 } : { ideal: 1280 }) : { ideal: 1920 },
      height: isMobile ? (isPortrait ? { ideal: 1280 } : { ideal: 720 }) : { ideal: 1080 }
    };

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      mediaStreamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      if (track) {
        const capabilities = track.getCapabilities?.() || {};
        setHasTorch(Boolean(capabilities.torch));
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (!videoRef.current) return;
          videoRef.current.play().then(() => {
            if (!videoRef.current) return;
            setIsStreaming(true);
            const vw = videoRef.current.videoWidth;
            const vh = videoRef.current.videoHeight;
            if (vw && vh) setAspectRatio(vw / vh);
          }).catch(e => {
            console.error("Play error:", e);
            setError("Could not autoplay camera. Please tap to enable.");
          });
        };
      }
    } catch (err) {
      console.warn("High-res camera failed, falling back to basic:", err);
      try {
        const fallbackConstraints = {
          facingMode: { ideal: mode },
          width: isMobile ? (isPortrait ? { ideal: 720 } : { ideal: 1280 }) : undefined,
          height: isMobile ? (isPortrait ? { ideal: 1280 } : { ideal: 720 }) : undefined,
        };
        const stream = await navigator.mediaDevices.getUserMedia({
          video: fallbackConstraints,
          audio: false,
        });
        mediaStreamRef.current = stream;
        const track = stream.getVideoTracks()[0];
        if (track) {
          const capabilities = track.getCapabilities?.() || {};
          setHasTorch(Boolean(capabilities.torch));
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            if (!videoRef.current) return;
            videoRef.current.play().then(() => {
              if (!videoRef.current) return;
              setIsStreaming(true);
              const vw = videoRef.current.videoWidth;
              const vh = videoRef.current.videoHeight;
              if (vw && vh) setAspectRatio(vw / vh);
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
    setIsTorchOn(false);
    setFacingMode(nextMode);
    startCamera(nextMode);
  };

  // Toggle Hardware Flashlight (Torch) with Screen Illumination Fallback
  const toggleTorch = async () => {
    const track = mediaStreamRef.current?.getVideoTracks()[0];
    const nextState = !isTorchOn;

    if (track) {
      try {
        const capabilities = track.getCapabilities?.() || {};
        if (capabilities.torch) {
          await track.applyConstraints({
            advanced: [{ torch: nextState }]
          });
        }
      } catch (err) {
        console.warn("Hardware torch error:", err);
      }
    }

    setIsTorchOn(nextState);
    safeSpeak(nextState ? "Flashlight on." : "Flashlight off.");
  };

  // Guidance Toggle (Switches Fullscreen in App + Starts Spoken Guide)
  const handleToggleGuidance = () => {
    const willBeActive = !isGuidanceActive;
    if (onToggleGuidance) {
      onToggleGuidance(willBeActive);
    }

    if (willBeActive) {
      const startMsg = "Guidance started. Point camera forward.";
      setSpokenSubtitles(startMsg);
      safeSpeak(startMsg);
    } else {
      if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    }
  };

  // Keyboard Escape listener to exit fullscreen guidance
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isGuidanceActive) {
        if (onToggleGuidance) onToggleGuidance(false);
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isGuidanceActive, onToggleGuidance]);

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
      const textToAnnounce = isAccepted ? data.text : "No clear text detected. Hold camera steady.";
      setRecognizedText(textToAnnounce);

      if (isAccepted) {
        let speakableText = data.text;
        if (speakableText.length > 200) {
          speakableText = speakableText.substring(0, 200) + "... and more.";
        }
        safeSpeak(speakableText);
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
    mountedRef.current = true;
    startCamera(facingMode);
    connectWebSocket();

    return () => {
      mountedRef.current = false;
      stopCamera();
      if (wsRef.current) wsRef.current.close();
      if (intervalRef.current) clearTimeout(intervalRef.current);
      // Close AudioContext to prevent resource leak
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(() => {});
        audioCtxRef.current = null;
      }
    };
  }, []);

  // 0. Proximity Sonar Ticking (Acoustic Parking Sensor for Real-Time 0ms Spatial Awareness)
  useEffect(() => {
    if (!isActive || !isGuidanceActive || !isSonarEnabled || isMuted) return;

    const dist = detections.closest_distance_m;
    if (dist === null || dist === undefined || dist > 1.8) return;

    // Rapid ticks as obstacle nears: <0.6m = 180ms, <1.1m = 380ms, <1.8m = 750ms
    let intervalMs = 750;
    let freq = 740;
    if (dist <= 0.6) {
      intervalMs = 180;
      freq = 1000;
    } else if (dist <= 1.1) {
      intervalMs = 380;
      freq = 880;
    }

    const now = Date.now();
    if (now - lastSonarPingRef.current >= intervalMs) {
      lastSonarPingRef.current = now;
      playSonarPing(freq, 0.04, 0.07);
    }
  }, [detections.closest_distance_m, isGuidanceActive, isSonarEnabled, isMuted, isActive, playSonarPing]);

  // 1. Local YOLO Immediate Obstacle Reflex Alert (< 1.2m or critical hazard)
  // DECOUPLED FROM GEMINI: Immediately interrupts & alerts on physical collision danger
  useEffect(() => {
    if (!isActive || !isGuidanceActive || isReadingText) return;
    if (!detections.narration || !detections.scene_changed) return;

    const now = Date.now();
    // 2.5s cooldown between consecutive reflex alerts to prevent stutter
    if (now - lastYoloAlertTimeRef.current < 2500) return;
    lastYoloAlertTimeRef.current = now;

    const narration = detections.narration;
    setSpokenSubtitles(narration);
    safeSpeak(narration);
  }, [detections.narration, detections.scene_changed, isGuidanceActive, isReadingText, isActive, safeSpeak]);

  // 2. Gemini High-Level Directional Guidance (Every ~3.5s when fresh)
  useEffect(() => {
    if (!isActive || !isGuidanceActive || isReadingText) {
      if (!isGuidanceActive && typeof window !== 'undefined' && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
      return;
    }

    const desc = detections.scene_description;
    if (!desc) return;

    // Trigger ONLY when fresh Gemini guidance arrives from backend
    if (detections.new_gemini_guidance && desc !== lastGeminiRef.current) {
      lastGeminiRef.current = desc;
      setSpokenSubtitles(desc);

      // If user is actively encountering a critical collision hazard (< 1.0m), prioritize safety reflex
      if (detections.is_critical_hazard) {
        return;
      }

      safeSpeak(desc);
    }
  }, [detections.new_gemini_guidance, detections.scene_description, detections.is_critical_hazard, isGuidanceActive, isReadingText, isActive, safeSpeak]);

  return (
    <div
      className={
        isGuidanceActive
          ? "fixed inset-0 z-50 bg-black flex flex-col w-full h-[100dvh] overflow-hidden select-none"
          : "relative flex-1 w-full h-full flex flex-col overflow-hidden"
      }
    >
      {/* Hidden canvas for frame capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* ─── Top Header Section ─── */}
      {!isGuidanceActive ? (
        /* Dashboard Card Header */
        <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/80 flex-shrink-0 bg-zinc-900/40">
          <div className="flex items-center gap-2 flex-1">
            <span className="text-lg">👁️</span>
            <div>
              <h2 className="text-xs md:text-sm font-bold text-zinc-100 flex items-center gap-2">
                Vision Assistant
                <span className="text-[0.6rem] px-2 py-0.5 rounded-full bg-zinc-900 text-zinc-400 border border-zinc-800 font-mono">
                  YOLO11m Edge
                </span>
              </h2>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={toggleTorch}
              className={`p-1.5 rounded-xl border transition-all text-xs flex items-center gap-1.5 px-3 cursor-pointer ${
                isTorchOn
                  ? 'bg-amber-400 text-black border-amber-300 font-bold'
                  : 'bg-zinc-900 hover:bg-zinc-800 border-zinc-800 text-zinc-300'
              }`}
              title={isTorchOn ? "Turn Flashlight OFF" : "Turn Flashlight ON"}
            >
              <span>🔦</span>
              <span className="hidden sm:inline font-semibold">{isTorchOn ? 'LIGHT ON' : 'Torch'}</span>
            </button>

            <button
              onClick={toggleCamera}
              className="p-1.5 rounded-xl bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 transition-all text-zinc-300 text-xs flex items-center gap-1.5 px-3 cursor-pointer"
              title="Switch Camera (Rear / Front)"
            >
              <span>🔄</span>
              <span className="hidden sm:inline font-semibold">{facingMode === 'environment' ? 'Rear' : 'Front'}</span>
            </button>

            <div className="flex items-center gap-1.5 ml-1 px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">
              <span className={`w-1.5 h-1.5 rounded-full ${isStreaming ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
              <span className={`text-[0.62rem] font-bold font-mono uppercase ${isConnected ? 'text-emerald-400' : 'text-zinc-500'}`}>
                {isConnected ? 'ONLINE' : 'OFFLINE'}
              </span>
            </div>
          </div>
        </div>
      ) : (
        /* Fullscreen Guidance Top Floating HUD Bar */
        <div className="absolute top-0 left-0 right-0 z-30 flex items-center justify-between p-2.5 sm:p-3 md:p-4 pt-safe bg-gradient-to-b from-black/95 via-black/75 to-transparent pointer-events-auto">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            <span className="text-[0.65rem] sm:text-xs md:text-sm font-bold text-emerald-300 bg-emerald-950/80 border border-emerald-500/40 px-2.5 sm:px-3 py-1 rounded-full shadow-lg font-mono">
              <span className="hidden sm:inline">● GEMINI </span>LIVE AI
            </span>
          </div>

          <div className="flex items-center gap-1.5 sm:gap-2">
            <button
              onClick={toggleTorch}
              className={`min-h-[44px] min-w-[44px] px-3 py-2 rounded-xl border text-xs font-semibold flex items-center justify-center gap-1.5 backdrop-blur-md transition-all cursor-pointer active:scale-95 ${
                isTorchOn
                  ? 'bg-amber-400 text-black border-amber-300 font-bold shadow-md'
                  : 'bg-zinc-900/90 hover:bg-zinc-800 border-zinc-700/80 text-zinc-200'
              }`}
              title={isTorchOn ? "Turn Flashlight OFF" : "Turn Flashlight ON"}
              aria-label={isTorchOn ? "Turn Flashlight OFF" : "Turn Flashlight ON"}
            >
              <span className="text-sm">🔦</span>
              <span className="hidden sm:inline">{isTorchOn ? 'LIGHT ON' : 'TORCH'}</span>
            </button>

            <button
              onClick={toggleCamera}
              className="min-h-[44px] min-w-[44px] px-3 py-2 rounded-xl bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-700/80 text-zinc-200 text-xs font-semibold flex items-center justify-center gap-1.5 backdrop-blur-md transition-all cursor-pointer active:scale-95"
              title="Switch Camera (Rear / Front)"
              aria-label="Switch Camera (Rear / Front)"
            >
              <span className="text-sm">🔄</span>
              <span className="hidden sm:inline">Flip</span>
            </button>

            <button
              onClick={handleReadText}
              disabled={isReadingText}
              className="min-h-[44px] min-w-[44px] px-3 py-2 rounded-xl bg-zinc-900/90 hover:bg-zinc-800 disabled:opacity-40 border border-zinc-700/80 text-zinc-200 text-xs font-semibold flex items-center justify-center gap-1.5 backdrop-blur-md transition-all cursor-pointer active:scale-95"
              title="Read Text in Front (OCR)"
              aria-label="Read Text in Front (OCR)"
            >
              <span className="text-sm">📖</span>
              <span className="hidden sm:inline">{isReadingText ? 'Reading...' : 'Read Text'}</span>
            </button>

            <button
              onClick={handleToggleGuidance}
              className="min-h-[44px] px-3.5 py-2 rounded-xl bg-red-600 hover:bg-red-500 border border-red-500 text-white text-xs font-bold flex items-center justify-center gap-1.5 transition-all shadow-md cursor-pointer active:scale-95"
              title="Stop Guidance and Return to Dashboard"
              aria-label="Stop Guidance and Return to Dashboard"
            >
              <span className="font-bold text-sm">✕</span>
              <span>EXIT</span>
            </button>
          </div>
        </div>
      )}

      {/* ─── Center-Top Gemini Spoken Guidance HUD Banner (Only in Fullscreen Guidance) ─── */}
      {isGuidanceActive && (detections.scene_description || spokenSubtitles) && (
        <div className="absolute top-14 sm:top-16 md:top-20 left-2 right-2 sm:left-4 sm:right-4 md:left-1/2 md:-translate-x-1/2 md:max-w-2xl z-20 pointer-events-none animate-fadeIn">
          <div className="bg-zinc-950/90 backdrop-blur-xl border border-zinc-800 rounded-2xl p-2.5 sm:p-3.5 md:p-4 shadow-xl flex items-start gap-2.5">
            <span className="text-xl sm:text-2xl flex-shrink-0">🧠</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-[0.6rem] sm:text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-400 font-mono">
                  Gemini Live Guidance:
                </span>
                <span className="text-[0.6rem] text-emerald-400 font-mono flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping"></span>
                  Guiding
                </span>
              </div>
              <p className="text-xs sm:text-sm md:text-base font-semibold text-zinc-100 leading-snug tracking-wide">
                {detections.scene_description || spokenSubtitles}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ─── Continuous Camera Viewport & Overlays (STABLE DOM NODE: NEVER UNMOUNTED) ─── */}
      <div className={`relative flex-1 w-full h-full min-h-0 flex items-center justify-center overflow-hidden ${isGuidanceActive ? 'bg-black' : 'bg-[#050510]'}`}>
        {error ? (
          <div className="absolute inset-0 flex items-center justify-center text-center text-red-400 p-8 z-10">
            <div>
              <p className="font-semibold text-sm">⚠️ {error}</p>
              <button
                onClick={() => startCamera(facingMode)}
                className="mt-4 px-6 py-2 bg-zinc-800 border border-zinc-700 rounded-xl text-white text-sm hover:bg-zinc-700 transition-colors cursor-pointer"
              >
                Retry Camera
              </button>
            </div>
          </div>
        ) : (
          <div className="relative flex items-center justify-center w-full h-full">
            <div
              className="relative flex items-center justify-center"
              style={{
                aspectRatio: aspectRatio || (4 / 3),
                maxWidth: '100%',
                maxHeight: '100%',
                width: 'auto',
                height: 'auto',
              }}
            >
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                aria-label="Live camera feed for AI object detection"
                onLoadedMetadata={(e) => {
                  if (e.target.videoWidth && e.target.videoHeight) {
                    setAspectRatio(e.target.videoWidth / e.target.videoHeight);
                  }
                }}
                className={`w-full h-full object-contain max-w-full max-h-full ${facingMode === 'user' ? '-scale-x-100' : ''}`}
              />

              {/* AI Bounding Box Overlays */}
              <div className={`absolute inset-0 pointer-events-none ${facingMode === 'user' ? '-scale-x-100' : ''}`}>
                {detections.objects?.map((obj, i) => {
                  const fs = detections.frame_size;
                  if (!fs) return null;
                  const style = {
                    left: `${(obj.box.x1 / fs.width) * 100}%`,
                    top: `${(obj.box.y1 / fs.height) * 100}%`,
                    width: `${((obj.box.x2 - obj.box.x1) / fs.width) * 100}%`,
                    height: `${((obj.box.y2 - obj.box.y1) / fs.height) * 100}%`,
                  };
                  const urgencyStyles = {
                    critical: {
                      border: 'border-red-500',
                      badge: 'bg-red-950/90 text-red-200 border-red-500 shadow-[0_0_16px_rgba(239,68,68,0.8)]',
                      glow: 'shadow-[0_0_20px_rgba(239,68,68,0.4)]',
                    },
                    warning: {
                      border: 'border-amber-400',
                      badge: 'bg-amber-950/90 text-amber-200 border-amber-400 shadow-[0_0_12px_rgba(245,158,11,0.7)]',
                      glow: 'shadow-[0_0_15px_rgba(245,158,11,0.3)]',
                    },
                    info: {
                      border: 'border-emerald-400/80',
                      badge: 'bg-emerald-950/90 text-emerald-200 border-emerald-400/70',
                      glow: '',
                    },
                    background: {
                      border: 'border-gray-400/60',
                      badge: 'bg-gray-950/90 text-gray-300 border-gray-400/50',
                      glow: '',
                    },
                  };
                  const stylePreset = urgencyStyles[obj.urgency] || urgencyStyles.info;
                  const distText = obj.distance_m ? `${obj.distance_m}m` : `~${obj.distance_steps}st`;
                  return (
                    <div
                      key={`obj-${i}`}
                      className={`absolute border-2 ${stylePreset.border} ${stylePreset.glow} rounded-lg transition-all duration-300 ease-linear pointer-events-none`}
                      style={style}
                    >
                      <span className={`absolute -top-6 left-0 ${stylePreset.badge} font-bold text-[0.62rem] md:text-[0.68rem] px-2 py-0.5 rounded-md whitespace-nowrap border backdrop-blur-md flex items-center gap-1 ${facingMode === 'user' ? '-scale-x-100' : ''}`}>
                        {obj.urgency === 'critical' ? '⚠️ ' : ''}
                        <span className="capitalize">{obj.label}</span>
                        <span className="font-mono text-[0.65rem] opacity-90">• {distText}</span>
                        {obj.position && <span className="opacity-70 text-[0.6rem]">({obj.position})</span>}
                      </span>
                    </div>
                  );
                })}

                {detections.emotions?.map((face, i) => {
                  const fs = detections.frame_size;
                  if (!fs) return null;
                  const style = {
                    left: `${(face.box.x / fs.width) * 100}%`,
                    top: `${(face.box.y / fs.height) * 100}%`,
                    width: `${(face.box.w / fs.width) * 100}%`,
                    height: `${(face.box.h / fs.height) * 100}%`,
                  };
                  return (
                    <div key={`emo-${i}`} className="absolute border-2 border-yellow-400 rounded-md transition-all duration-300 ease-linear" style={style}>
                      <span className={`absolute -top-5 left-0 bg-yellow-500 text-black font-bold text-[0.62rem] px-1.5 py-0.5 rounded whitespace-nowrap ${facingMode === 'user' ? '-scale-x-100' : ''}`}>
                        {face.emotion} {Math.round(face.confidence)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ─── Floating OCR Result (When OCR runs) ─── */}
      {recognizedText && (
        <div className={
          isGuidanceActive
            ? "absolute inset-x-3 sm:inset-x-4 top-20 sm:top-24 z-40 bg-zinc-950/95 backdrop-blur-xl border border-zinc-700 rounded-2xl p-3 sm:p-4 shadow-2xl animate-fadeIn"
            : "px-4 py-3 border-t border-zinc-800 bg-zinc-900/60 flex-shrink-0"
        }>
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-xs font-bold text-zinc-300 uppercase tracking-wider flex items-center gap-1.5 font-mono">
              <span>📖</span> Detected Text (OCR)
            </span>
            <button
              onClick={() => setRecognizedText(null)}
              className="text-zinc-400 hover:text-white text-xs font-bold px-2 py-1 bg-zinc-800 rounded-lg cursor-pointer transition-colors"
            >
              ✕ Close
            </button>
          </div>
          <p className="text-xs sm:text-sm text-zinc-100 whitespace-pre-wrap font-medium leading-relaxed max-h-48 overflow-y-auto">
            {recognizedText}
          </p>
        </div>
      )}

      {/* ─── Bottom Section (Conditional: Normal Dashboard Controls OR Fullscreen Floating Bar) ─── */}
      {isGuidanceActive ? (
        /* Fullscreen Floating Obstacle Radar Bar */
        <div className="absolute bottom-0 left-0 right-0 z-30 p-2.5 sm:p-3 md:p-4 pb-safe bg-gradient-to-t from-black/95 via-black/80 to-transparent flex flex-col gap-1.5 sm:gap-2 pointer-events-auto">
          {/* Obstacle Radar Chips */}
          {detections.objects && detections.objects.length > 0 && (
            <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
              <span className="text-[0.62rem] text-zinc-500 font-bold uppercase flex-shrink-0 font-mono mr-0.5">
                Radar:
              </span>
              {detections.objects.slice(0, 6).map((obj, i) => {
                const isCritical = obj.urgency === 'critical';
                const chipColor = isCritical
                  ? 'bg-red-500/30 border-red-500 text-red-200 animate-bounce'
                  : obj.urgency === 'warning'
                    ? 'bg-amber-500/20 border-amber-500/40 text-amber-200'
                    : 'bg-zinc-900/90 border-zinc-700/80 text-zinc-200';
                return (
                  <span
                    key={i}
                    className={`text-[0.68rem] sm:text-xs px-2.5 py-1 rounded-lg border font-semibold whitespace-nowrap flex items-center gap-1.5 backdrop-blur-md ${chipColor}`}
                  >
                    {isCritical && <span>⚠️</span>}
                    <span>{obj.label}</span>
                    <span className="font-mono text-[0.65rem] sm:text-[0.7rem] opacity-80 font-bold">
                      {obj.distance_m ? `${obj.distance_m}m` : `~${obj.distance_steps}st`}
                    </span>
                    <span className="text-[0.6rem] opacity-60">({obj.position || 'ahead'})</span>
                  </span>
                );
              })}
            </div>
          )}

          {/* Subtitle & Mute Bar */}
          <div className="flex items-center justify-between gap-2 pt-1.5 border-t border-zinc-800/80">
            <div className="flex items-center gap-1.5 text-xs text-zinc-300 min-w-0 flex-1">
              <span className="text-sm flex-shrink-0">🔊</span>
              <span className="truncate italic text-[0.7rem] sm:text-xs text-zinc-200">
                {spokenSubtitles || 'Point phone forward — AI observing path...'}
              </span>
            </div>

            <div className="flex items-center gap-1.5 flex-shrink-0">
              <button
                onClick={() => setIsSonarEnabled(!isSonarEnabled)}
                className={`min-h-[38px] px-2.5 py-1.5 rounded-lg text-xs font-semibold border cursor-pointer flex items-center gap-1 transition-colors active:scale-95 ${
                  isSonarEnabled
                    ? 'bg-zinc-800 text-white border-zinc-700 shadow-sm'
                    : 'bg-zinc-900/80 text-zinc-500 border-zinc-800'
                }`}
                title="Toggle Proximity Sonar Beep"
              >
                <span>📡</span>
                <span>{isSonarEnabled ? 'Sonar ON' : 'Sonar OFF'}</span>
              </button>

              <button
                onClick={() => {
                  const nextMute = !isMuted;
                  setIsMuted(nextMute);
                  if (nextMute && typeof window !== 'undefined' && 'speechSynthesis' in window) {
                    window.speechSynthesis.cancel();
                  }
                }}
                className="min-h-[38px] px-2.5 py-1.5 rounded-lg bg-zinc-900/80 hover:bg-zinc-800 text-zinc-200 text-xs font-medium border border-zinc-800 cursor-pointer flex items-center gap-1 active:scale-95 transition-colors"
              >
                <span>{isMuted ? '🔇' : '🔊'}</span>
                <span className="hidden sm:inline">{isMuted ? 'Unmute' : 'Mute'}</span>
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* Normal Dashboard Controls */
        <>
          {/* Gemini Scene Description Bar */}
          {detections.scene_description && (
            <div className="px-4 py-2 border-t border-zinc-800 bg-zinc-900/60 flex-shrink-0">
              <p className="text-xs text-zinc-300 leading-relaxed font-medium">
                <span className="text-zinc-400 font-bold">🧠 AI Guide:</span> {detections.scene_description}
              </p>
            </div>
          )}

          {/* Detected Objects Chips in Card */}
          {detections.objects && detections.objects.length > 0 && (
            <div className="px-3 py-1.5 border-t border-zinc-800/80 flex items-center gap-1.5 overflow-x-auto flex-shrink-0 bg-black/40">
              <span className="text-[0.62rem] text-zinc-500 uppercase font-mono font-bold mr-0.5">Objects:</span>
              {detections.objects.slice(0, 5).map((obj, i) => {
                const tagColors = {
                  critical: 'bg-red-500/20 text-red-300 border-red-500/40',
                  warning:  'bg-amber-500/20 text-amber-300 border-amber-500/40',
                  info:     'bg-zinc-800/80 text-zinc-300 border-zinc-700/80',
                  background: 'bg-zinc-900 text-zinc-400 border-zinc-800',
                };
                const tc = tagColors[obj.urgency] || tagColors.info;
                return (
                  <span key={`tag-${i}`} className={`text-[0.65rem] font-semibold px-2 py-0.5 ${tc} rounded-lg whitespace-nowrap border flex items-center gap-1`}>
                    {obj.urgency === 'critical' && <span>⚠️</span>}
                    <span>{obj.label}</span>
                    <span className="font-mono opacity-80">{obj.distance_m ? `${obj.distance_m}m` : ''}</span>
                  </span>
                );
              })}
            </div>
          )}

          {/* Primary Action Buttons */}
          <div className="p-3 border-t border-zinc-800/80 bg-[#0e0e11] flex-shrink-0 flex items-center gap-2.5">
            <button
              onClick={handleToggleGuidance}
              className="flex-1 py-3 px-4 rounded-xl bg-white hover:bg-zinc-200 text-zinc-950 font-bold text-xs sm:text-sm shadow-sm flex items-center justify-center gap-2 transition-all cursor-pointer active:scale-95"
            >
              <span className="text-base sm:text-lg">🔊</span>
              <span className="tracking-wide">GUIDE ON (FULLSCREEN AI)</span>
            </button>

            <button
              onClick={handleReadText}
              disabled={isReadingText || !isStreaming}
              className="px-4 py-3 rounded-xl bg-zinc-900 hover:bg-zinc-800 disabled:opacity-40 border border-zinc-700/80 text-zinc-200 font-semibold text-xs sm:text-sm flex items-center justify-center gap-1.5 transition-all cursor-pointer active:scale-95"
            >
              <span>📖</span>
              <span className="hidden sm:inline">{isReadingText ? 'READING...' : 'READ TEXT'}</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default CameraFeed;
