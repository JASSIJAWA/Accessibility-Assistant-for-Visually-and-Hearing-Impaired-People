import { useRef, useEffect, useState, useCallback } from 'react';

const getHearingWsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/hearing`;
};

function HearingAssistant() {
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioStreamRef = useRef(null);
  const recognitionRef = useRef(null);

  const [isListening, setIsListening] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [interimText, setInterimText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [inputText, setInputText] = useState('');
  const [isProcessingAudio, setIsProcessingAudio] = useState(false);

  const transcriptEndRef = useRef(null);
  const isListeningRef = useRef(false);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript, interimText]);

  // WebSocket setup
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

        setIsProcessingAudio(false);

        // If backend returned a transcription from audio
        if (data.transcription && data.transcription.trim()) {
          setTranscript(prev => {
            const last = prev[prev.length - 1];
            if (last && last.text === data.transcription.trim()) return prev;
            return [...prev.slice(-49), {
              text: data.transcription.trim(),
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }];
          });
        }

        if (data.summary || data.tone) {
          setAnalysis(data);

          // Urgent haptic alert
          if (data.is_urgent && typeof navigator !== 'undefined' && navigator.vibrate) {
            navigator.vibrate([250, 100, 250]);
          }
        }
      } catch (e) {
        console.error("Hearing message parse error", e);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      ws.close();
      stopListening();
    };
  }, []);

  // Send text to backend
  const sendText = useCallback((text) => {
    if (!text.trim()) return;
    const trimmed = text.trim();

    setTranscript(prev => [...prev.slice(-49), {
      text: trimmed,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'transcript', text: trimmed }));
    }
    setInputText('');
  }, []);

  // Send audio chunk to backend
  const sendAudioChunk = useCallback((blob, mimeType) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) return;
    setIsProcessingAudio(true);

    const reader = new FileReader();
    reader.onloadend = () => {
      const base64data = reader.result;
      wsRef.current.send(JSON.stringify({
        action: 'audio',
        data: base64data,
        mime_type: mimeType || 'audio/webm'
      }));
    };
    reader.readAsDataURL(blob);
  }, []);

  // Start microphone listening via MediaRecorder (works on 100% of mobile browsers)
  const startListening = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : '';

      const options = mimeType ? { mimeType } : {};
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 2000) {
          sendAudioChunk(event.data, recorder.mimeType);
        }
      };

      // Record in 4-second continuous segments
      recorder.start(4000);
      setIsListening(true);
      isListeningRef.current = true;

      // Optional: Also attach Web Speech API for immediate live text preview if available
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        try {
          const recognition = new SpeechRecognition();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = 'en-US';
          recognitionRef.current = recognition;

          recognition.onresult = (e) => {
            let interim = '';
            for (let i = e.resultIndex; i < e.results.length; i++) {
              if (e.results[i].isFinal) {
                const text = e.results[i][0].transcript.trim();
                if (text) sendText(text);
                setInterimText('');
              } else {
                interim += e.results[i][0].transcript;
              }
            }
            setInterimText(interim);
          };

          recognition.onend = () => {
            if (isListeningRef.current) {
              try { recognition.start(); } catch (err) { /* ignore */ }
            }
          };

          recognition.onerror = () => { /* fallback to MediaRecorder silently */ };
          recognition.start();
        } catch (e) {
          console.log("Web Speech API fallback to MediaRecorder");
        }
      }

    } catch (err) {
      console.error("Microphone error", err);
      setError(`Microphone access denied: ${err.message || 'Please allow mic permissions in browser'}`);
      setIsListening(false);
      isListeningRef.current = false;
    }
  };

  const stopListening = () => {
    isListeningRef.current = false;
    setIsListening(false);
    setInterimText('');

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch (e) {}
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach(t => t.stop());
      audioStreamRef.current = null;
    }
    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
      recognitionRef.current = null;
    }
  };

  const resetAll = () => {
    stopListening();
    setTranscript([]);
    setAnalysis(null);
    setInterimText('');
    setError(null);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    }
  };

  // Tone styling
  const toneColor = {
    urgent:    'text-red-300 bg-red-500/15 border-red-500/40',
    alarmed:   'text-red-300 bg-red-500/15 border-red-500/40',
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
    <div className="flex flex-col h-full text-sm">

      {/* ── Top Bar ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.07] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg">🎙️</span>
          <div>
            <h2 className="font-semibold text-gray-100 text-xs md:text-sm">Hearing & Speech Assistant</h2>
            <p className="text-[0.65rem] text-white/40">Real-time Speech-to-Text & AI Tone Analysis</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {isConnected ? (
            <span className="text-[0.65rem] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              ● Connected
            </span>
          ) : (
            <span className="text-[0.65rem] px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20">
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
        <div className="mx-3 mt-2 px-3.5 py-2.5 rounded-xl bg-red-500/25 border border-red-500/50 text-red-200 text-xs font-bold flex items-center gap-2 animate-bounce flex-shrink-0 shadow-[0_0_20px_rgba(239,68,68,0.3)]">
          <span className="text-xl">🚨</span>
          <div>
            <div>URGENT SITUATION DETECTED</div>
            <div className="text-[0.65rem] font-normal text-red-300">Speaker is issuing an alert or distress call</div>
          </div>
        </div>
      )}

      {/* ── AI Analysis Summary Card ── */}
      <div className="mx-3 mt-2.5 flex-shrink-0">
        {analysis?.summary ? (
          <div className={`p-3 rounded-xl border backdrop-blur-md transition-all ${currentToneClass}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xl">{analysis.tone_emoji}</span>
                <span className="font-bold capitalize text-xs tracking-wide">
                  {analysis.tone} Tone
                </span>
              </div>
              <span className="text-[0.6rem] font-medium px-2 py-0.5 rounded-md bg-white/10 opacity-70">
                Gemini AI Summary
              </span>
            </div>

            <p className="text-xs font-medium leading-relaxed opacity-95">
              {analysis.summary}
            </p>

            {analysis.key_points && analysis.key_points.length > 0 && (
              <div className="mt-2.5 pt-2 border-t border-white/10">
                <p className="text-[0.65rem] uppercase tracking-wider opacity-60 font-semibold mb-1">Key Takeaways:</p>
                <ul className="space-y-1">
                  {analysis.key_points.map((pt, i) => (
                    <li key={i} className="text-xs opacity-85 flex items-start gap-1.5">
                      <span className="text-purple-400 font-bold">•</span>
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ) : (
          <div className="p-3 rounded-xl border border-white/5 bg-white/[0.02] text-center">
            <p className="text-xs text-white/40">
              {isListening ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                  Listening to conversation... AI summary will generate automatically
                </span>
              ) : (
                'Tap "Start Listening" or use presets below to analyze speech'
              )}
            </p>
          </div>
        )}
      </div>

      {/* ── Conversation Presets (Quick Test) ── */}
      <div className="mx-3 mt-2 flex-shrink-0">
        <p className="text-[0.65rem] text-white/30 uppercase tracking-wider mb-1 px-1">Quick Test Scenarios:</p>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => sendText("Watch out! There is a fast moving car coming right towards you!")}
            className="text-[0.65rem] px-2 py-1 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/20 transition-colors"
          >
            🚨 Emergency Warning
          </button>
          <button
            onClick={() => sendText("Excuse me, could you please tell me where the nearest metro station is?")}
            className="text-[0.65rem] px-2 py-1 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/20 transition-colors"
          >
            😊 Asking Directions
          </button>
          <button
            onClick={() => sendText("I've been waiting here for two hours, this service is completely unacceptable!")}
            className="text-[0.65rem] px-2 py-1 rounded-lg bg-orange-500/10 hover:bg-orange-500/20 text-orange-300 border border-orange-500/20 transition-colors"
          >
            😠 Angry Complaint
          </button>
        </div>
      </div>

      {/* ── Live Transcript Stream ── */}
      <div className="flex-1 overflow-y-auto px-3.5 py-3 space-y-2 min-h-0">
        {transcript.length === 0 && !interimText && (
          <div className="flex flex-col items-center justify-center h-full text-white/20 text-xs text-center">
            <span className="text-3xl mb-2">💬</span>
            <p>Spoken dialogue will appear here in real-time</p>
          </div>
        )}

        {transcript.map((line, i) => (
          <div key={i} className="flex gap-2.5 items-start p-2 rounded-lg bg-white/[0.02] border border-white/5">
            <span className="text-[0.6rem] text-purple-400/80 font-mono mt-0.5 flex-shrink-0 font-semibold">
              {line.time}
            </span>
            <p className="text-xs text-gray-200 leading-relaxed font-sans">{line.text}</p>
          </div>
        ))}

        {interimText && (
          <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
            <p className="text-xs text-purple-200 italic leading-relaxed animate-pulse">
              🎙️ {interimText}…
            </p>
          </div>
        )}

        {isProcessingAudio && (
          <div className="flex items-center gap-2 text-[0.65rem] text-purple-300/60 italic px-2">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-ping"></span>
            Transcribing audio segment...
          </div>
        )}

        <div ref={transcriptEndRef} />
      </div>

      {/* ── Error Display ── */}
      {error && (
        <div className="mx-3 mb-2 px-3 py-2 rounded-xl bg-red-500/15 border border-red-500/30 text-red-300 text-xs flex-shrink-0">
          ⚠️ {error}
        </div>
      )}

      {/* ── Controls & Text Input Bar ── */}
      <div className="p-3 border-t border-white/[0.07] bg-[#0c0c1e]/60 flex-shrink-0 space-y-2">
        {/* Mic Toggle Button */}
        {isListening ? (
          <button
            onClick={stopListening}
            className="w-full py-2.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-semibold text-xs transition-all flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(239,68,68,0.2)]"
          >
            <span className="w-2.5 h-2.5 rounded-full bg-red-400 animate-ping" />
            Stop Microphone Listening
          </button>
        ) : (
          <button
            onClick={startListening}
            disabled={!isConnected}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-purple-600/30 to-indigo-600/30 hover:from-purple-600/40 hover:to-indigo-600/40 border border-purple-500/40 text-purple-200 font-semibold text-xs transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-[0_0_15px_rgba(168,85,247,0.15)]"
          >
            🎙️ Start Microphone Listening
          </button>
        )}

        {/* Text Input for Manual Input or Simulation */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendText(inputText);
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type a spoken phrase or test sentence..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-gray-200 placeholder-white/30 focus:outline-none focus:border-purple-500/50"
          />
          <button
            type="submit"
            disabled={!inputText.trim()}
            className="px-3.5 py-2 bg-purple-500/20 hover:bg-purple-500/30 disabled:opacity-30 border border-purple-500/30 text-purple-300 rounded-xl text-xs font-semibold transition-colors"
          >
            Send
          </button>
        </form>
      </div>

    </div>
  );
}

export default HearingAssistant;
