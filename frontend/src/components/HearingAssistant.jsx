import { useRef, useEffect, useState, useCallback } from 'react';

const getHearingWsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/hearing`;
};

function HearingAssistant() {
  const wsRef = useRef(null);
  const recognitionRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordTimerRef = useRef(null);
  const processingTimeoutRef = useRef(null);

  const [isConnected, setIsConnected] = useState(false);
  const [isLiveListening, setIsLiveListening] = useState(false);
  const [isRecordingAudio, setIsRecordingAudio] = useState(false);
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const [transcript, setTranscript] = useState([]);
  const [interimText, setInterimText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [inputText, setInputText] = useState('');

  const transcriptEndRef = useRef(null);
  const keepListeningRef = useRef(false);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript, interimText]);

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

        // Clear processing spinner
        setIsProcessing(false);
        if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);

        // If backend returned a transcription from audio
        if (data.transcription && data.transcription.trim()) {
          setTranscript(prev => {
            const trimmed = data.transcription.trim();
            const last = prev[prev.length - 1];
            if (last && last.text === trimmed) return prev;
            return [...prev.slice(-49), {
              text: trimmed,
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }];
          });
        }

        if (data.summary || data.tone) {
          setAnalysis(data);

          // Haptic alert on emergency
          if (data.is_urgent && typeof navigator !== 'undefined' && navigator.vibrate) {
            navigator.vibrate([250, 100, 250]);
          }
        }
      } catch (e) {
        setIsProcessing(false);
        console.error("Message error:", e);
      }
    };

    ws.onerror = () => {
      setIsProcessing(false);
      setError('Connection error with AI backend');
    };

    ws.onclose = () => {
      setIsProcessing(false);
      setIsConnected(false);
    };

    return () => {
      keepListeningRef.current = false;
      stopLiveListening();
      stopVoiceRecording();
      ws.close();
    };
  }, []);

  // Send plain transcript text to backend
  const sendText = useCallback((text) => {
    if (!text || !text.trim()) return;
    const trimmed = text.trim();

    setTranscript(prev => [...prev.slice(-49), {
      text: trimmed,
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }]);

    setIsProcessing(true);
    if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
    processingTimeoutRef.current = setTimeout(() => setIsProcessing(false), 12000);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'transcript', text: trimmed }));
    }
    setInputText('');
  }, []);

  // ─────────────────────────────────────────────────────────────────────────────
  // 1. LIVE CAPTIONS (Web Speech API)
  // ─────────────────────────────────────────────────────────────────────────────
  const startLiveListening = () => {
    setError(null);
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Live browser speech recognition not supported in this browser. Please use the "Record Voice Clip" button below or Chrome.');
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognitionRef.current = recognition;
      keepListeningRef.current = true;

      recognition.onstart = () => {
        setIsLiveListening(true);
      };

      recognition.onresult = (e) => {
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) {
            const text = e.results[i][0].transcript.trim();
            if (text) {
              sendText(text);
            }
            setInterimText('');
          } else {
            interim += e.results[i][0].transcript;
          }
        }
        setInterimText(interim);
      };

      recognition.onerror = (e) => {
        console.warn('SpeechRecognition error:', e.error);
        if (e.error === 'not-allowed') {
          setError('Microphone permission denied. Please allow microphone access in your browser.');
          setIsLiveListening(false);
          keepListeningRef.current = false;
        } else if (e.error === 'network') {
          setError('Browser speech network error. Use the "Record Voice Clip" button below.');
          setIsLiveListening(false);
          keepListeningRef.current = false;
        }
      };

      recognition.onend = () => {
        if (keepListeningRef.current) {
          try {
            recognition.start();
          } catch (err) {
            setIsLiveListening(false);
          }
        } else {
          setIsLiveListening(false);
          setInterimText('');
        }
      };

      // MUST be called synchronously within user click!
      recognition.start();
    } catch (err) {
      console.error('Speech recognition start failed:', err);
      setError(`Failed to start live speech: ${err.message}. Try "Record Voice Clip" instead.`);
      setIsLiveListening(false);
    }
  };

  const stopLiveListening = () => {
    keepListeningRef.current = false;
    setIsLiveListening(false);
    setInterimText('');
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (e) {}
      recognitionRef.current = null;
    }
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // 2. VOICE CLIP RECORDING (MediaRecorder -> Gemini Audio AI)
  // ─────────────────────────────────────────────────────────────────────────────
  const startVoiceRecording = async () => {
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
        if (audioBlob.size > 1000) {
          setIsProcessing(true);
          if (processingTimeoutRef.current) clearTimeout(processingTimeoutRef.current);
          processingTimeoutRef.current = setTimeout(() => setIsProcessing(false), 15000);

          const reader = new FileReader();
          reader.onloadend = () => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify({
                action: 'audio',
                data: reader.result,
                mime_type: recorder.mimeType || 'audio/webm'
              }));
            }
          };
          reader.readAsDataURL(audioBlob);
        }
      };

      recorder.start();
      setIsRecordingAudio(true);

      recordTimerRef.current = setInterval(() => {
        setRecordSeconds(s => s + 1);
      }, 1000);

    } catch (err) {
      console.error('Mic record error:', err);
      setError(`Microphone error: ${err.message || 'Permission denied'}`);
      setIsRecordingAudio(false);
    }
  };

  const stopVoiceRecording = () => {
    setIsRecordingAudio(false);
    if (recordTimerRef.current) clearInterval(recordTimerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {}
    }
  };

  const resetAll = () => {
    stopLiveListening();
    stopVoiceRecording();
    setTranscript([]);
    setAnalysis(null);
    setInterimText('');
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
    <div className="flex flex-col h-full text-sm">

      {/* ── Top Bar ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.07] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-lg">🎙️</span>
          <div>
            <h2 className="font-semibold text-gray-100 text-xs md:text-sm">Hearing & Speech Assistant</h2>
            <p className="text-[0.65rem] text-white/40">Real-time Speech Recognition & Tone Analysis</p>
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
            <div className="text-[0.65rem] font-normal text-red-300">Speaker is sounding an alarm, danger, or emergency!</div>
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
              {isLiveListening ? (
                <span className="flex items-center justify-center gap-2 text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                  Listening continuously... Speak near the microphone
                </span>
              ) : isRecordingAudio ? (
                <span className="flex items-center justify-center gap-2 text-red-400">
                  <span className="w-2 h-2 rounded-full bg-red-400 animate-ping"></span>
                  Recording audio clip ({recordSeconds}s)... Tap "Stop & Analyze" when done
                </span>
              ) : (
                'Choose a voice option below or tap a quick scenario to analyze speech'
              )}
            </p>
          </div>
        )}
      </div>

      {/* ── Quick Test Presets (One Tap) ── */}
      <div className="mx-3 mt-2 flex-shrink-0">
        <p className="text-[0.65rem] text-white/30 uppercase tracking-wider mb-1 px-1">Quick Scenarios (One-Tap Test):</p>
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => sendText("Watch out! There is a fast moving car coming right towards you!")}
            className="text-[0.65rem] px-2.5 py-1.5 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-300 border border-red-500/20 transition-colors flex items-center gap-1"
          >
            <span>🚨</span> Emergency Alert
          </button>
          <button
            onClick={() => sendText("Excuse me, could you please tell me where the nearest metro station is?")}
            className="text-[0.65rem] px-2.5 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-300 border border-emerald-500/20 transition-colors flex items-center gap-1"
          >
            <span>😊</span> Asking Help
          </button>
          <button
            onClick={() => sendText("I have been waiting here for an hour, this service is completely unacceptable!")}
            className="text-[0.65rem] px-2.5 py-1.5 rounded-lg bg-orange-500/10 hover:bg-orange-500/20 text-orange-300 border border-orange-500/20 transition-colors flex items-center gap-1"
          >
            <span>😠</span> Angry Complaint
          </button>
          <button
            onClick={() => sendText("The meeting is scheduled for tomorrow morning at 10 AM in conference room B.")}
            className="text-[0.65rem] px-2.5 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-300 border border-sky-500/20 transition-colors flex items-center gap-1"
          >
            <span>💼</span> Normal Info
          </button>
        </div>
      </div>

      {/* ── Live Transcript Stream ── */}
      <div className="flex-1 overflow-y-auto px-3.5 py-3 space-y-2 min-h-0">
        {transcript.length === 0 && !interimText && (
          <div className="flex flex-col items-center justify-center h-full text-white/20 text-xs text-center py-6">
            <span className="text-3xl mb-2">💬</span>
            <p>Spoken words and transcripts will appear here in real time</p>
          </div>
        )}

        {transcript.map((line, i) => (
          <div key={i} className="flex gap-2.5 items-start p-2.5 rounded-xl bg-white/[0.03] border border-white/5">
            <span className="text-[0.6rem] text-purple-400 font-mono mt-0.5 flex-shrink-0 font-semibold">
              {line.time}
            </span>
            <p className="text-xs text-gray-200 leading-relaxed font-sans">{line.text}</p>
          </div>
        ))}

        {interimText && (
          <div className="p-2.5 rounded-xl bg-purple-500/10 border border-purple-500/25">
            <p className="text-xs text-purple-200 italic leading-relaxed animate-pulse">
              🎙️ {interimText}…
            </p>
          </div>
        )}

        {isProcessing && (
          <div className="flex items-center gap-2 text-xs text-purple-300 bg-purple-500/10 border border-purple-500/20 p-2 rounded-xl">
            <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping"></span>
            <span>Gemini AI is analyzing conversation & tone...</span>
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

      {/* ── Voice & Text Controls ── */}
      <div className="p-3 border-t border-white/[0.07] bg-[#0c0c1e]/70 flex-shrink-0 space-y-2.5">

        {/* Dual Mode Buttons: Live Captions OR Record Voice Note */}
        <div className="grid grid-cols-2 gap-2">
          {/* Mode 1: Live Captions */}
          {isLiveListening ? (
            <button
              onClick={stopLiveListening}
              className="py-2.5 px-3 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-red-300 font-semibold text-xs transition-all flex items-center justify-center gap-1.5"
            >
              <span className="w-2 h-2 rounded-full bg-red-400 animate-ping" />
              Stop Captions
            </button>
          ) : (
            <button
              onClick={startLiveListening}
              disabled={isRecordingAudio || !isConnected}
              className="py-2.5 px-3 rounded-xl bg-purple-600/25 hover:bg-purple-600/35 border border-purple-500/40 text-purple-200 font-semibold text-xs transition-all disabled:opacity-40 flex items-center justify-center gap-1.5"
            >
              <span>🎙️</span> Live Captions
            </button>
          )}

          {/* Mode 2: Push-to-Talk Record Audio Clip */}
          {isRecordingAudio ? (
            <button
              onClick={stopVoiceRecording}
              className="py-2.5 px-3 rounded-xl bg-red-600/30 hover:bg-red-600/40 border border-red-500/50 text-red-200 font-semibold text-xs transition-all flex items-center justify-center gap-1.5 animate-pulse"
            >
              <span className="w-2 h-2 rounded-full bg-red-400 animate-ping" />
              Stop & Send ({recordSeconds}s)
            </button>
          ) : (
            <button
              onClick={startVoiceRecording}
              disabled={isLiveListening || !isConnected}
              className="py-2.5 px-3 rounded-xl bg-indigo-600/25 hover:bg-indigo-600/35 border border-indigo-500/40 text-indigo-200 font-semibold text-xs transition-all disabled:opacity-40 flex items-center justify-center gap-1.5"
            >
              <span>🔊</span> Record Audio Clip
            </button>
          )}
        </div>

        {/* Text Input Bar */}
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
            placeholder="Or type a spoken phrase to analyze..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-gray-200 placeholder-white/30 focus:outline-none focus:border-purple-500/50"
          />
          <button
            type="submit"
            disabled={!inputText.trim()}
            className="px-3.5 py-2 bg-purple-500/25 hover:bg-purple-500/35 disabled:opacity-30 border border-purple-500/30 text-purple-300 rounded-xl text-xs font-semibold transition-colors"
          >
            Send
          </button>
        </form>
      </div>

    </div>
  );
}

export default HearingAssistant;
