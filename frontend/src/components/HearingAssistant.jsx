import { useRef, useEffect, useState, useCallback } from 'react';

const getHearingWsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/hearing`;
};

// ─── Encode raw Float32 PCM to 16kHz mono WAV ────────────────────────────────
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // SubChunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
  view.setUint16(22, 1, true);  // NumChannels (1 = Mono)
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // ByteRate
  view.setUint16(32, 2, true);  // BlockAlign
  view.setUint16(34, 16, true); // BitsPerSample
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function HearingAssistant() {
  const wsRef = useRef(null);
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const processorRef = useRef(null);
  const pcmChunksRef = useRef([]);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);
  const recordTimerRef = useRef(null);
  const processingTimeoutRef = useRef(null);
  const transcriptEndRef = useRef(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0); // 0 to 100 for visualizer

  const [transcript, setTranscript] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [replyText, setReplyText] = useState('');

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  // WebSocket Connection
  useEffect(() => {
    let reconnectTimeout = null;

    const connect = () => {
      const ws = new WebSocket(getHearingWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === 'reset') return;

          setIsProcessing(false);
          if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);

          // Transcription from Gemini or Fallback Whisper
          if (data.transcription && data.transcription.trim()) {
            const text = data.transcription.trim();
            setTranscript(prev => {
              const last = prev[prev.length - 1];
              if (last && last.text === text) return prev;
              return [...prev.slice(-49), {
                text,
                tone: data.tone || 'neutral',
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
              }];
            });
          }

          // Summary and Tone
          if (data.summary || data.tone) {
            setAnalysis(data);

            // Emergency vibration
            if (data.is_urgent && typeof navigator !== 'undefined' && navigator.vibrate) {
              try {
                navigator.vibrate([300, 150, 300, 150, 400]);
              } catch (e) {}
            }
          }
        } catch (e) {
          setIsProcessing(false);
          console.error("Message parse error:", e);
        }
      };

      ws.onerror = () => {
        setIsProcessing(false);
        setError('Backend WebSocket disconnected. Retrying...');
      };

      ws.onclose = () => {
        setIsProcessing(false);
        setIsConnected(false);
        reconnectTimeout = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      stopRecording();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Send WAV Audio Blob to Backend Gemini Multimodal AI
  const sendAudioBlob = (wavBlob) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      setError('Cannot send audio: Server disconnected.');
      return;
    }

    setIsProcessing(true);
    if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
    processingTimeoutRef.current = setTimeout(() => setIsProcessing(false), 15000);

    const reader = new FileReader();
    reader.onloadend = () => {
      wsRef.current.send(JSON.stringify({
        action: 'audio',
        data: reader.result,
        mime_type: 'audio/wav'
      }));
    };
    reader.readAsDataURL(wavBlob);
  };

  // Send plain text (e.g. for simulations or typed dialogue)
  const sendText = useCallback((text) => {
    if (!text || !text.trim()) return;
    const trimmed = text.trim();

    setTranscript(prev => [...prev.slice(-49), {
      text: trimmed,
      tone: 'user',
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    }]);

    setIsProcessing(true);
    if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
    processingTimeoutRef.current = setTimeout(() => setIsProcessing(false), 10000);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'transcript', text: trimmed }));
    }
  }, []);

  // ─────────────────────────────────────────────────────────────────────────────
  // High-Fidelity 16kHz WAV Audio Recording with Web Audio API
  // ─────────────────────────────────────────────────────────────────────────────
  const startRecording = async () => {
    setError(null);
    pcmChunksRef.current = [];
    setRecordSeconds(0);
    setAudioLevel(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      mediaStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);

      // Analyser for visual volume meter
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;
      source.connect(analyser);

      const updateLevel = () => {
        if (!analyserRef.current) return;
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        setAudioLevel(Math.min(100, Math.round((avg / 128) * 100)));
        animFrameRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();

      // ScriptProcessorNode captures raw PCM Float32
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsRecording(true);
      recordTimerRef.current = setInterval(() => {
        setRecordSeconds(s => s + 1);
      }, 1000);

    } catch (err) {
      console.error('Microphone error:', err);
      setError(`Microphone error: ${err.message || 'Permission denied. Please grant microphone access.'}`);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (!isRecording && !mediaStreamRef.current) return;
    setIsRecording(false);

    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    setAudioLevel(0);

    // Stop audio tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }

    // Disconnect Web Audio processor
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
      } catch (e) {}
      processorRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {}
      audioContextRef.current = null;
    }

    // Merge PCM chunks and encode to 16kHz WAV
    const chunks = pcmChunksRef.current;
    if (chunks.length > 0) {
      let totalLength = 0;
      for (let i = 0; i < chunks.length; i++) {
        totalLength += chunks[i].length;
      }

      // Minimum ~0.4s of audio (6400 samples at 16kHz)
      if (totalLength > 6400) {
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (let i = 0; i < chunks.length; i++) {
          merged.set(chunks[i], offset);
          offset += chunks[i].length;
        }

        const wavBlob = encodeWAV(merged, 16000);
        if (wavBlob.size > 1000) {
          sendAudioBlob(wavBlob);
        }
      }
    }
    pcmChunksRef.current = [];
  };

  // Two-way communication: Deaf user types reply, phone speaks it aloud
  const speakReply = (textToSpeak) => {
    const text = textToSpeak || replyText;
    if (!text || !text.trim()) return;

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.95;
      window.speechSynthesis.speak(utterance);
    }
    setReplyText('');
  };

  const resetAll = () => {
    stopRecording();
    setTranscript([]);
    setAnalysis(null);
    setError(null);
    setIsProcessing(false);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    }
  };

  const toneColor = {
    urgent:    'text-red-300 bg-red-500/15 border-red-500/40',
    alarmed:   'text-red-300 bg-red-500/15 border-red-500/40',
    emergency: 'text-red-300 bg-red-500/15 border-red-500/40',
    angry:     'text-orange-300 bg-orange-500/15 border-orange-500/40',
    worried:   'text-yellow-300 bg-yellow-500/15 border-yellow-500/40',
    confused:  'text-yellow-200 bg-yellow-400/15 border-yellow-400/40',
    sad:       'text-blue-300 bg-blue-500/15 border-blue-500/40',
    happy:     'text-emerald-300 bg-emerald-500/15 border-emerald-500/40',
    excited:   'text-green-300 bg-green-500/15 border-green-500/40',
    calm:      'text-sky-300 bg-sky-500/15 border-sky-500/40',
    neutral:   'text-gray-300 bg-gray-500/15 border-gray-500/40',
  };

  const currentToneClass = analysis ? (toneColor[analysis.tone] || toneColor.neutral) : toneColor.neutral;

  return (
    <div className="flex flex-col h-full text-sm bg-transparent">

      {/* ── Top Header ── */}
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-white/[0.08] flex-shrink-0 bg-white/[0.02]">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-lg">
            🎙️
          </div>
          <div>
            <h2 className="font-bold text-gray-100 text-xs md:text-sm flex items-center gap-2">
              Hearing & Speech Assistant
              <span className="text-[0.6rem] font-bold px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/35 font-mono">
                Gemini Audio AI
              </span>
            </h2>
            <p className="text-[0.65rem] text-white/40 font-medium">100% accurate verbatim captions, tone detection & two-way speech</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isConnected ? (
            <span className="text-[0.62rem] px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/25 font-bold font-mono">
              ● ONLINE
            </span>
          ) : (
            <span className="text-[0.62rem] px-2.5 py-1 rounded-full bg-red-500/10 text-red-400 border border-red-500/25 font-bold font-mono">
              ● CONNECTING
            </span>
          )}
          <button
            onClick={resetAll}
            className="text-[0.62rem] px-2.5 py-1 rounded-lg bg-white/[0.05] hover:bg-white/[0.1] text-white/60 hover:text-white transition-colors border border-white/10 uppercase font-mono cursor-pointer"
          >
            Clear
          </button>
        </div>
      </div>

      {/* ── Urgent Alert Banner ── */}
      {analysis?.is_urgent && (
        <div className="mx-3 mt-2.5 px-3.5 py-2.5 rounded-xl bg-red-500/20 border border-red-500/50 text-red-200 text-xs font-bold flex items-center gap-3 animate-bounce flex-shrink-0 shadow-[0_0_25px_rgba(239,68,68,0.35)]">
          <span className="text-2xl flex-shrink-0">🚨</span>
          <div>
            <div className="tracking-wide uppercase font-black text-red-300">URGENT SITUATION OR SAFETY HAZARD DETECTED</div>
            <div className="text-[0.68rem] font-normal text-red-200/90 mt-0.5">Speaker is warning of immediate danger, sounding an alert, or crying for help!</div>
          </div>
        </div>
      )}

      {/* ── AI Tone & Cognitive Summary Card (Constrained Height for Mobile) ── */}
      <div className="mx-2.5 sm:mx-3 mt-2 sm:mt-2.5 flex-shrink-0">
        {analysis?.summary ? (
          <div className={`p-3 sm:p-3.5 rounded-2xl border backdrop-blur-md transition-all shadow-xl max-h-40 sm:max-h-52 overflow-y-auto ${currentToneClass}`}>
            <div className="flex items-center justify-between mb-1.5 sm:mb-2 sticky top-0">
              <div className="flex items-center gap-1.5 sm:gap-2">
                <span className="text-xl sm:text-2xl">{analysis.tone_emoji || '🎙️'}</span>
                <span className="font-bold capitalize text-xs sm:text-sm tracking-wide">
                  {analysis.tone} Tone Detected
                </span>
              </div>
              <span className="text-[0.58rem] sm:text-[0.62rem] font-bold px-2 py-0.5 rounded-md bg-white/15 opacity-90 uppercase tracking-wider font-mono">
                Cognitive Understanding
              </span>
            </div>

            <p className="text-xs sm:text-sm font-semibold leading-relaxed opacity-95">
              {analysis.summary}
            </p>

            {analysis.key_points && analysis.key_points.length > 0 && (
              <div className="mt-2 pt-1.5 border-t border-white/15">
                <p className="text-[0.6rem] sm:text-[0.62rem] uppercase tracking-wider opacity-70 font-bold mb-1 font-mono">Key Takeaways:</p>
                <ul className="space-y-1">
                  {analysis.key_points.map((pt, i) => (
                    <li key={i} className="text-xs opacity-90 flex items-start gap-1.5">
                      <span className="text-purple-300 font-bold">•</span>
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="p-3 sm:p-3.5 rounded-2xl border border-white/[0.08] bg-white/[0.02] text-center shadow-inner">
            <p className="text-xs text-white/50">
              {isRecording ? (
                <span className="flex items-center justify-center gap-2 text-red-400 font-bold animate-pulse">
                  <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-red-400 animate-ping"></span>
                  Listening ({recordSeconds}s)... Tap "Stop & Transcribe"
                </span>
              ) : isProcessing ? (
                <span className="flex items-center justify-center gap-2 text-purple-300 font-bold">
                  <span className="w-2 h-2 sm:w-2.5 sm:h-2.5 rounded-full bg-purple-400 animate-ping"></span>
                  Gemini Audio AI transcribing speech...
                </span>
              ) : (
                'Tap the microphone button below and speak'
              )}
            </p>
          </div>
        )}
      </div>

      {/* ── Audio Visualizer Spectrum (Visible While Recording) ── */}
      {isRecording && (
        <div className="mx-3 mt-2 px-3.5 py-2.5 rounded-xl bg-purple-950/40 border border-purple-500/40 flex items-center justify-between gap-3 flex-shrink-0 shadow-lg">
          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
            <span className="text-xs font-mono text-purple-200 font-bold">
              REC {String(Math.floor(recordSeconds / 60)).padStart(2, '0')}:{String(recordSeconds % 60).padStart(2, '0')}
            </span>
          </div>

          {/* 16-Bar Spectrum Equalizer */}
          <div className="flex-1 flex items-center justify-center gap-1 h-6 px-2">
            {[35, 60, 45, 80, 95, 65, 85, 50, 75, 90, 40, 70, 85, 55, 75, 45].map((baseH, idx) => {
              const dynamicH = Math.min(100, Math.max(15, (audioLevel / 100) * baseH * 1.3));
              return (
                <div
                  key={idx}
                  className="w-1.5 rounded-full bg-gradient-to-t from-purple-500 via-indigo-400 to-emerald-400 transition-all duration-75"
                  style={{ height: `${dynamicH}%` }}
                />
              );
            })}
          </div>

          <span className="text-xs font-mono text-emerald-400 font-bold flex-shrink-0">
            {audioLevel}%
          </span>
        </div>
      )}

      {/* ── Live Transcripts / Captions Feed ── */}
      <div className="flex-1 overflow-y-auto px-3.5 py-3 space-y-2.5 min-h-0">
        {transcript.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-white/20 text-xs text-center py-8">
            <div className="w-12 h-12 rounded-2xl bg-white/[0.03] border border-white/5 flex items-center justify-center text-2xl mb-2">
              💬
            </div>
            <p className="font-semibold text-white/40">Verbatim transcripts & speech captions will appear here</p>
            <p className="text-[0.68rem] text-white/25 mt-1 font-mono">Tap the microphone button below and speak</p>
          </div>
        )}

        {transcript.map((line, i) => {
          // Check if text has acoustic bracket sounds e.g. [horn], [siren]
          const parts = line.text.split(/(\[[^\]]+\])/g);
          return (
            <div key={i} className="flex gap-3 items-start p-3 rounded-2xl bg-white/[0.03] border border-white/[0.06] hover:border-white/[0.12] transition-colors shadow-sm">
              <span className="text-[0.62rem] text-purple-300 font-mono mt-0.5 flex-shrink-0 font-bold bg-purple-500/15 border border-purple-500/25 px-2 py-0.5 rounded-md">
                {line.time}
              </span>
              <p className="text-xs md:text-sm text-gray-100 leading-relaxed font-sans font-medium flex-1">
                {parts.map((part, pIdx) => {
                  if (part.startsWith('[') && part.endsWith(']')) {
                    return (
                      <span key={pIdx} className="inline-block mx-1 px-1.5 py-0.5 rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono text-[0.75rem] font-bold">
                        {part}
                      </span>
                    );
                  }
                  return part;
                })}
              </p>
            </div>
          );
        })}

        {isProcessing && (
          <div className="flex items-center gap-2.5 text-xs text-purple-300 bg-purple-500/15 border border-purple-500/30 p-3 rounded-2xl shadow-lg">
            <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-ping"></span>
            <span className="font-semibold">Gemini Multimodal AI transcribing speech & analyzing emotion...</span>
          </div>
        )}

        <div ref={transcriptEndRef} />
      </div>

      {/* ── Error Notification ── */}
      {error && (
        <div className="mx-3 mb-2 px-3 py-2 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs flex-shrink-0 font-medium">
          ⚠️ {error}
        </div>
      )}

      {/* ── Two-Way Communication: Quick Speech Responses (Compact on Mobile) ── */}
      <div className="px-2.5 sm:px-3.5 py-1.5 sm:py-2 border-t border-white/[0.06] bg-black/30 flex-shrink-0 flex items-center gap-1.5 overflow-x-auto scrollbar-none">
        <span className="text-[0.6rem] sm:text-[0.62rem] text-white/40 font-bold uppercase font-mono flex-shrink-0">Quick TTS:</span>
        <button
          onClick={() => speakReply("I am hard of hearing. Please speak clearly.")}
          className="text-[0.65rem] sm:text-[0.68rem] px-2.5 py-1 rounded-lg bg-white/[0.05] hover:bg-white/[0.1] text-white/80 hover:text-white whitespace-nowrap border border-white/10 transition-colors cursor-pointer active:scale-95"
        >
          🔊 "I am hard of hearing"
        </button>
        <button
          onClick={() => speakReply("Could you please repeat that?")}
          className="text-[0.65rem] sm:text-[0.68rem] px-2.5 py-1 rounded-lg bg-white/[0.05] hover:bg-white/[0.1] text-white/80 hover:text-white whitespace-nowrap border border-white/10 transition-colors cursor-pointer active:scale-95"
        >
          🔊 "Please repeat that"
        </button>
        <button
          onClick={() => speakReply("Yes, I understand. Thank you.")}
          className="text-[0.65rem] sm:text-[0.68rem] px-2.5 py-1 rounded-lg bg-white/[0.05] hover:bg-white/[0.1] text-white/80 hover:text-white whitespace-nowrap border border-white/10 transition-colors cursor-pointer active:scale-95"
        >
          🔊 "Yes, I understand"
        </button>
      </div>

      {/* ── Main Controls (Microphone & Custom Type) ── */}
      <div className="p-2.5 sm:p-3.5 border-t border-white/[0.08] bg-[#080816] flex-shrink-0 space-y-2 sm:space-y-2.5">

        {/* Big Action Button */}
        {isRecording ? (
          <button
            onClick={stopRecording}
            className="w-full py-3 sm:py-3.5 rounded-2xl bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 border border-red-400 text-white font-extrabold text-xs sm:text-sm transition-all flex items-center justify-center gap-2 shadow-[0_0_25px_rgba(239,68,68,0.5)] animate-pulse cursor-pointer active:scale-95"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-white animate-ping" />
            <span className="tracking-wide">STOP & TRANSCRIBE ({recordSeconds}s)</span>
          </button>
        ) : (
          <button
            onClick={startRecording}
            disabled={!isConnected}
            className="w-full py-3 sm:py-3.5 rounded-2xl bg-gradient-to-r from-purple-600 via-indigo-600 to-violet-600 hover:from-purple-500 hover:to-indigo-500 border border-purple-400/80 text-white font-extrabold text-xs sm:text-sm transition-all disabled:opacity-40 flex items-center justify-center gap-2 shadow-[0_0_25px_rgba(168,85,247,0.4)] cursor-pointer group active:scale-95"
          >
            <span className="text-base group-hover:scale-125 transition-transform">🎙️</span>
            <span className="tracking-wide">TAP TO SPEAK (VOICE CAPTIONS)</span>
          </button>
        )}

        {/* Custom Text & TTS Reply Bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendText(replyText);
            setReplyText('');
          }}
          className="flex gap-1.5 sm:gap-2"
        >
          <input
            type="text"
            value={replyText}
            onChange={(e) => setReplyText(e.target.value)}
            placeholder="Type spoken dialogue or reply..."
            className="flex-1 bg-white/[0.04] border border-white/[0.1] rounded-xl px-3 py-2 sm:px-3.5 sm:py-2.5 text-xs text-gray-100 placeholder-white/30 focus:outline-none focus:border-purple-400 focus:bg-white/[0.07] transition-all"
          />
          <button
            type="submit"
            disabled={!replyText.trim()}
            className="px-3 py-2 sm:px-3.5 sm:py-2.5 bg-purple-500/20 hover:bg-purple-500/35 disabled:opacity-30 border border-purple-500/35 text-purple-300 rounded-xl text-xs font-bold transition-colors cursor-pointer active:scale-95"
            title="Analyze as incoming speech"
          >
            Analyze
          </button>
          <button
            type="button"
            onClick={() => speakReply()}
            disabled={!replyText.trim()}
            className="px-3 py-2 sm:px-3.5 sm:py-2.5 bg-emerald-500/20 hover:bg-emerald-500/35 disabled:opacity-30 border border-emerald-500/35 text-emerald-300 rounded-xl text-xs font-bold transition-colors flex items-center gap-1 cursor-pointer active:scale-95"
            title="Speak aloud through phone speaker"
          >
            <span>🔊</span> Speak
          </button>
        </form>

      </div>

    </div>
  );
}

export default HearingAssistant;
