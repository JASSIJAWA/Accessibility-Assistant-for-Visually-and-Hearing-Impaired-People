import { useRef, useEffect, useState, useCallback } from 'react';

const getHearingWsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/hearing`;
};

function HearingAssistant() {
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordTimerRef = useRef(null);
  const continuousTimerRef = useRef(null);
  const processingTimeoutRef = useRef(null);
  const transcriptEndRef = useRef(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isContinuous, setIsContinuous] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

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

        // Transcription from Whisper
        if (data.transcription && data.transcription.trim()) {
          const text = data.transcription.trim();
          setTranscript(prev => {
            const last = prev[prev.length - 1];
            if (last && last.text === text) return prev;
            return [...prev.slice(-49), {
              text,
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }];
          });
        }

        // Summary and Tone from Gemini
        if (data.summary || data.tone) {
          setAnalysis(data);

          // Emergency vibration
          if (data.is_urgent && typeof navigator !== 'undefined' && navigator.vibrate) {
            navigator.vibrate([300, 150, 300]);
          }
        }
      } catch (e) {
        setIsProcessing(false);
        console.error("Message error:", e);
      }
    };

    ws.onerror = () => {
      setIsProcessing(false);
      setError('Connection to AI backend failed. Make sure backend is running.');
    };

    ws.onclose = () => {
      setIsProcessing(false);
      setIsConnected(false);
    };

    return () => {
      stopRecording();
      if (continuousTimerRef.current) clearInterval(continuousTimerRef.current);
      ws.close();
    };
  }, []);

  // Send raw audio blob to backend Whisper engine
  const sendAudioBlob = (blob, mimeType) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    setIsProcessing(true);

    if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
    processingTimeoutRef.current = setTimeout(() => setIsProcessing(false), 12000);

    const reader = new FileReader();
    reader.onloadend = () => {
      wsRef.current.send(JSON.stringify({
        action: 'audio',
        data: reader.result,
        mime_type: mimeType || 'audio/webm'
      }));
    };
    reader.readAsDataURL(blob);
  };

  // Send plain text (e.g. for simulations or typed dialogue)
  const sendText = useCallback((text) => {
    if (!text || !text.trim()) return;
    const trimmed = text.trim();

    setTranscript(prev => [...prev.slice(-49), {
      text: trimmed,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);

    setIsProcessing(true);
    if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
    processingTimeoutRef.current = setTimeout(() => setIsProcessing(false), 10000);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'transcript', text: trimmed }));
    }
  }, []);

  // ─────────────────────────────────────────────────────────────────────────────
  // Push-to-Talk Recording (Guaranteed to work on all mobile devices)
  // ─────────────────────────────────────────────────────────────────────────────
  const startRecording = async () => {
    setError(null);
    audioChunksRef.current = [];
    setRecordSeconds(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : '';

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        if (recordTimerRef.current) clearInterval(recordTimerRef.current);

        const audioBlob = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        if (audioBlob.size > 800) {
          sendAudioBlob(audioBlob, recorder.mimeType);
        }
      };

      recorder.start();
      setIsRecording(true);

      recordTimerRef.current = setInterval(() => {
        setRecordSeconds(s => s + 1);
      }, 1000);

    } catch (err) {
      console.error('Microphone error:', err);
      setError(`Microphone access error: ${err.message || 'Permission denied. Please allow mic permissions in browser settings.'}`);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {}
    }
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
    <div className="flex flex-col h-full text-sm bg-[#0e0e22]">

      {/* ── Top Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.08] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-xl">🎙️</span>
          <div>
            <h2 className="font-bold text-gray-100 text-sm md:text-base flex items-center gap-2">
              Hearing & Speech Assistant
              <span className="text-[0.6rem] font-semibold px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30">
                Whisper AI + Gemini
              </span>
            </h2>
            <p className="text-[0.65rem] text-white/40">Real-time captions, tone alerts & two-way speech</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isConnected ? (
            <span className="text-[0.65rem] px-2.5 py-1 rounded-full bg-emerald-500/15 text-emerald-400 border border-emerald-500/25 font-semibold">
              ● Online
            </span>
          ) : (
            <span className="text-[0.65rem] px-2.5 py-1 rounded-full bg-red-500/15 text-red-400 border border-red-500/25 font-semibold">
              ● Connecting...
            </span>
          )}
          <button
            onClick={resetAll}
            className="text-[0.65rem] px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-white/50 hover:text-white/80 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* ── Urgent Alert Banner ── */}
      {analysis?.is_urgent && (
        <div className="mx-3 mt-2 px-3.5 py-2.5 rounded-xl bg-red-500/25 border border-red-500/50 text-red-200 text-xs font-bold flex items-center gap-2.5 animate-bounce flex-shrink-0 shadow-[0_0_20px_rgba(239,68,68,0.3)]">
          <span className="text-2xl">🚨</span>
          <div>
            <div className="tracking-wide">EMERGENCY / URGENT SITUATION DETECTED</div>
            <div className="text-[0.68rem] font-normal text-red-300 mt-0.5">Speaker is sounding an alert, warning of danger, or asking for help!</div>
          </div>
        </div>
      )}

      {/* ── AI Tone & Summary Card ── */}
      <div className="mx-3 mt-2.5 flex-shrink-0">
        {analysis?.summary ? (
          <div className={`p-3.5 rounded-2xl border backdrop-blur-md transition-all shadow-lg ${currentToneClass}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-2xl">{analysis.tone_emoji}</span>
                <span className="font-bold capitalize text-sm tracking-wide">
                  {analysis.tone} Tone Detected
                </span>
              </div>
              <span className="text-[0.62rem] font-semibold px-2 py-0.5 rounded-md bg-white/10 opacity-80">
                Cognitive Summary
              </span>
            </div>

            <p className="text-xs md:text-sm font-medium leading-relaxed opacity-95">
              {analysis.summary}
            </p>

            {analysis.key_points && analysis.key_points.length > 0 && (
              <div className="mt-2.5 pt-2 border-t border-white/10">
                <p className="text-[0.65rem] uppercase tracking-wider opacity-60 font-bold mb-1">Key Takeaways:</p>
                <ul className="space-y-1">
                  {analysis.key_points.map((pt, i) => (
                    <li key={i} className="text-xs opacity-90 flex items-start gap-2">
                      <span className="text-purple-400 font-bold">•</span>
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="p-3.5 rounded-2xl border border-white/5 bg-white/[0.02] text-center">
            <p className="text-xs text-white/40">
              {isRecording ? (
                <span className="flex items-center justify-center gap-2 text-red-400 font-semibold">
                  <span className="w-2.5 h-2.5 rounded-full bg-red-400 animate-ping"></span>
                  Listening to voice ({recordSeconds}s)... Tap "Stop & Transcribe" when finished speaking
                </span>
              ) : isProcessing ? (
                <span className="flex items-center justify-center gap-2 text-purple-300 font-semibold">
                  <span className="w-2.5 h-2.5 rounded-full bg-purple-400 animate-ping"></span>
                  Transcribing with Whisper AI & generating tone analysis...
                </span>
              ) : (
                'Tap the microphone button below to capture speech, or test with quick scenarios'
              )}
            </p>
          </div>
        )}
      </div>

      {/* ── Quick Test Scenarios (Instant Demonstration) ── */}
      <div className="mx-3 mt-2 flex-shrink-0">
        <p className="text-[0.65rem] text-white/30 uppercase tracking-wider mb-1 px-1">Quick Scenarios (One-Tap Test):</p>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => sendText("Watch out! There is a fast moving car coming right towards you!")}
            className="text-[0.68rem] px-2.5 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/25 transition-colors flex items-center gap-1 font-medium"
          >
            <span>🚨</span> Emergency Alert
          </button>
          <button
            onClick={() => sendText("Excuse me, could you please tell me where the nearest pharmacy is located?")}
            className="text-[0.68rem] px-2.5 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/25 transition-colors flex items-center gap-1 font-medium"
          >
            <span>😊</span> Asking Help
          </button>
          <button
            onClick={() => sendText("I have been waiting here for two hours, this is completely unacceptable and ridiculous!")}
            className="text-[0.68rem] px-2.5 py-1.5 rounded-lg bg-orange-500/10 hover:bg-orange-500/20 text-orange-300 border border-orange-500/25 transition-colors flex items-center gap-1 font-medium"
          >
            <span>😠</span> Angry Complaint
          </button>
          <button
            onClick={() => sendText("The project evaluation is scheduled for tomorrow morning in the computer science lab.")}
            className="text-[0.68rem] px-2.5 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-300 border border-sky-500/25 transition-colors flex items-center gap-1 font-medium"
          >
            <span>💼</span> Normal Info
          </button>
        </div>
      </div>

      {/* ── Live Transcripts / Captions Feed ── */}
      <div className="flex-1 overflow-y-auto px-3.5 py-3 space-y-2.5 min-h-0">
        {transcript.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-white/20 text-xs text-center py-8">
            <span className="text-4xl mb-2">💬</span>
            <p className="font-medium">Live speech transcripts and captions will appear here</p>
            <p className="text-[0.65rem] text-white/15 mt-1">Tap the purple microphone below and speak naturally</p>
          </div>
        )}

        {transcript.map((line, i) => (
          <div key={i} className="flex gap-3 items-start p-3 rounded-xl bg-white/[0.03] border border-white/5 hover:border-white/10 transition-colors">
            <span className="text-[0.65rem] text-purple-400 font-mono mt-0.5 flex-shrink-0 font-bold bg-purple-500/10 px-1.5 py-0.5 rounded">
              {line.time}
            </span>
            <p className="text-xs md:text-sm text-gray-100 leading-relaxed font-sans font-medium">{line.text}</p>
          </div>
        ))}

        {isProcessing && (
          <div className="flex items-center gap-2 text-xs text-purple-300 bg-purple-500/10 border border-purple-500/25 p-2.5 rounded-xl">
            <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping"></span>
            <span>Faster-Whisper transcribing speech & analyzing sentiment...</span>
          </div>
        )}

        <div ref={transcriptEndRef} />
      </div>

      {/* ── Error Notification ── */}
      {error && (
        <div className="mx-3 mb-2 px-3 py-2 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs flex-shrink-0">
          ⚠️ {error}
        </div>
      )}

      {/* ── Two-Way Communication: Quick Speech Responses ── */}
      <div className="px-3 py-2 border-t border-white/[0.05] bg-black/20 flex-shrink-0 flex items-center gap-1.5 overflow-x-auto">
        <span className="text-[0.65rem] text-white/40 font-semibold uppercase flex-shrink-0">Quick Reply (TTS):</span>
        <button
          onClick={() => speakReply("I am hard of hearing. Please speak clearly.")}
          className="text-[0.65rem] px-2 py-1 rounded-md bg-white/5 hover:bg-white/10 text-white/70 whitespace-nowrap border border-white/5"
        >
          🔊 "I am hard of hearing"
        </button>
        <button
          onClick={() => speakReply("Could you please repeat that?")}
          className="text-[0.65rem] px-2 py-1 rounded-md bg-white/5 hover:bg-white/10 text-white/70 whitespace-nowrap border border-white/5"
        >
          🔊 "Please repeat that"
        </button>
        <button
          onClick={() => speakReply("Yes, I understand. Thank you.")}
          className="text-[0.65rem] px-2 py-1 rounded-md bg-white/5 hover:bg-white/10 text-white/70 whitespace-nowrap border border-white/5"
        >
          🔊 "Yes, I understand"
        </button>
      </div>

      {/* ── Main Controls (Microphone & Custom Type) ── */}
      <div className="p-3 border-t border-white/[0.08] bg-[#0c0c1e] flex-shrink-0 space-y-2.5">

        {/* Big Action Button */}
        {isRecording ? (
          <button
            onClick={stopRecording}
            className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500 border border-red-400 text-white font-bold text-sm transition-all flex items-center justify-center gap-2.5 shadow-[0_0_25px_rgba(239,68,68,0.4)] animate-pulse"
          >
            <span className="w-3 h-3 rounded-full bg-white animate-ping" />
            <span>Stop & Transcribe ({recordSeconds}s)</span>
          </button>
        ) : (
          <button
            onClick={startRecording}
            disabled={!isConnected}
            className="w-full py-3.5 rounded-2xl bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 border border-purple-400 text-white font-bold text-sm transition-all disabled:opacity-40 flex items-center justify-center gap-2.5 shadow-[0_0_25px_rgba(168,85,247,0.3)] cursor-pointer"
          >
            <span className="text-base">🎙️</span>
            <span>Tap to Speak (Voice Captions)</span>
          </button>
        )}

        {/* Custom Text & TTS Reply Bar */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendText(replyText);
            setReplyText('');
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={replyText}
            onChange={(e) => setReplyText(e.target.value)}
            placeholder="Type spoken dialogue or a response to speak..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3.5 py-2.5 text-xs text-gray-200 placeholder-white/30 focus:outline-none focus:border-purple-500/60"
          />
          <button
            type="submit"
            disabled={!replyText.trim()}
            className="px-3 py-2 bg-purple-500/20 hover:bg-purple-500/30 disabled:opacity-30 border border-purple-500/30 text-purple-300 rounded-xl text-xs font-semibold transition-colors"
            title="Analyze as incoming speech"
          >
            Analyze
          </button>
          <button
            type="button"
            onClick={() => speakReply()}
            disabled={!replyText.trim()}
            className="px-3 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 disabled:opacity-30 border border-emerald-500/30 text-emerald-300 rounded-xl text-xs font-semibold transition-colors flex items-center gap-1"
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
