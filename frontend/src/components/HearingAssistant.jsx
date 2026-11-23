import { useRef, useEffect, useState, useCallback } from 'react';

const getHearingWsUrl = () => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/hearing`;
};

function HearingAssistant() {
  const wsRef = useRef(null);
  const recognitionRef = useRef(null);

  const [isListening, setIsListening] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcript, setTranscript] = useState([]);    // array of {text, time}
  const [interimText, setInterimText] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const transcriptEndRef = useRef(null);

  // ── Scroll transcript to bottom ───────────────────────────────────────────
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [transcript]);

  // ── WebSocket connection ───────────────────────────────────────────────────
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
        setAnalysis(data);

        // Urgent alert — flash the page or use vibration
        if (data.is_urgent && navigator.vibrate) {
          navigator.vibrate([200, 100, 200]);
        }
      } catch (e) { /* ignore */ }
    };

    ws.onerror = () => setError('Connection error');
    ws.onclose = () => setIsConnected(false);

    return () => ws.close();
  }, []);

  // ── Send transcript line to backend ───────────────────────────────────────
  const sendTranscript = useCallback((text) => {
    if (wsRef.current?.readyState === WebSocket.OPEN && text.trim()) {
      wsRef.current.send(JSON.stringify({ action: 'transcript', text }));
    }
  }, []);

  // ── Web Speech API ────────────────────────────────────────────────────────
  const startListening = useCallback(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser. Use Chrome.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognitionRef.current = recognition;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => {
      setIsListening(false);
      setInterimText('');
    };

    recognition.onerror = (e) => {
      if (e.error !== 'no-speech') setError(`Mic error: ${e.error}`);
    };

    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          const finalText = result[0].transcript.trim();
          if (finalText) {
            setTranscript(prev => [...prev.slice(-49), {
              text: finalText,
              time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }]);
            sendTranscript(finalText);
          }
          setInterimText('');
        } else {
          interim += result[0].transcript;
        }
      }
      setInterimText(interim);
    };

    recognition.start();
  }, [sendTranscript]);

  const stopListening = useCallback(() => {
    recognitionRef.current?.stop();
    setIsListening(false);
    setInterimText('');
  }, []);

  const resetAll = useCallback(() => {
    stopListening();
    setTranscript([]);
    setAnalysis(null);
    setInterimText('');
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: 'reset' }));
    }
  }, [stopListening]);

  // ── Tone color map ────────────────────────────────────────────────────────
  const toneColor = {
    urgent:  'text-red-400 bg-red-500/10 border-red-500/30',
    angry:   'text-orange-400 bg-orange-500/10 border-orange-500/30',
    worried: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
    confused:'text-yellow-300 bg-yellow-400/10 border-yellow-400/30',
    sad:     'text-blue-400 bg-blue-500/10 border-blue-500/30',
    happy:   'text-green-400 bg-green-500/10 border-green-500/30',
    excited: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
    calm:    'text-sky-400 bg-sky-500/10 border-sky-500/30',
    neutral: 'text-gray-400 bg-gray-500/10 border-gray-500/30',
  };

  const tc = analysis ? (toneColor[analysis.tone] || toneColor.neutral) : toneColor.neutral;

  return (
    <div className="flex flex-col h-full text-sm">

      {/* ── Header ── */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.07] flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-base">🎙️</span>
          <span className="font-semibold text-gray-200 text-xs md:text-sm">Hearing Assistant</span>
          {isConnected
            ? <span className="text-[0.6rem] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">● Connected</span>
            : <span className="text-[0.6rem] px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/20">● Offline</span>
          }
        </div>
        <button
          onClick={resetAll}
          className="text-[0.65rem] px-2 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-white/40 hover:text-white/70 transition-colors"
        >Reset</button>
      </div>

      {/* ── Urgent alert banner ── */}
      {analysis?.is_urgent && (
        <div className="mx-3 mt-2 px-3 py-2 rounded-xl bg-red-500/20 border border-red-500/40 text-red-300 text-xs font-semibold flex items-center gap-2 animate-pulse flex-shrink-0">
          🚨 Urgent situation detected!
        </div>
      )}

      {/* ── AI Analysis panel ── */}
      {analysis?.summary ? (
        <div className={`mx-3 mt-2 px-3 py-2.5 rounded-xl border flex-shrink-0 ${tc}`}>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-base">{analysis.tone_emoji}</span>
            <span className="font-semibold capitalize text-xs">{analysis.tone} tone</span>
            <span className="ml-auto text-[0.6rem] opacity-50">AI Summary</span>
          </div>
          <p className="text-xs leading-relaxed opacity-90">{analysis.summary}</p>
          {analysis.key_points?.length > 0 && (
            <ul className="mt-2 space-y-0.5">
              {analysis.key_points.map((pt, i) => (
                <li key={i} className="text-[0.65rem] opacity-75 flex items-start gap-1">
                  <span className="mt-0.5">•</span><span>{pt}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <div className="mx-3 mt-2 px-3 py-2.5 rounded-xl border border-white/5 bg-white/[0.03] flex-shrink-0">
          <p className="text-xs text-white/30 text-center">
            {isListening ? 'Listening… AI summary will appear here' : 'Start listening to analyze conversation'}
          </p>
        </div>
      )}

      {/* ── Live transcript ── */}
      <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5 min-h-0">
        {transcript.length === 0 && !interimText && (
          <p className="text-xs text-white/20 text-center pt-4">No transcript yet</p>
        )}
        {transcript.map((line, i) => (
          <div key={i} className="flex gap-2 items-start">
            <span className="text-[0.55rem] text-white/20 mt-0.5 flex-shrink-0 font-mono">{line.time}</span>
            <p className="text-xs text-gray-300 leading-relaxed">{line.text}</p>
          </div>
        ))}
        {interimText && (
          <p className="text-xs text-white/30 italic leading-relaxed">{interimText}…</p>
        )}
        <div ref={transcriptEndRef} />
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="mx-3 mb-2 px-3 py-2 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-xs flex-shrink-0">
          ⚠️ {error}
        </div>
      )}

      {/* ── Controls ── */}
      <div className="px-3 py-3 border-t border-white/[0.07] flex-shrink-0">
        {isListening ? (
          <button
            onClick={stopListening}
            className="w-full py-2.5 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/30 text-red-300 font-semibold text-xs transition-all flex items-center justify-center gap-2"
          >
            <span className="inline-block w-2 h-2 rounded-full bg-red-400 animate-pulse" />
            Stop Listening
          </button>
        ) : (
          <button
            onClick={startListening}
            disabled={!isConnected}
            className="w-full py-2.5 rounded-xl bg-purple-500/20 hover:bg-purple-500/30 border border-purple-500/30 text-purple-300 font-semibold text-xs transition-all disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            🎙️ Start Listening
          </button>
        )}
      </div>
    </div>
  );
}

export default HearingAssistant;
