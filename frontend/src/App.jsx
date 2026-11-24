import { useState, useCallback, useEffect } from 'react';
import CameraFeed from './components/CameraFeed';
import SpatialRadarPanel from './components/SpatialRadarPanel';
import AlertPanel from './components/AlertPanel';
import HearingAssistant from './components/HearingAssistant';

function App() {
  const [visionData, setVisionData] = useState(null);
  const [activeTab, setActiveTab] = useState('vision'); // 'vision' | 'hearing'
  const [isGuidanceActive, setIsGuidanceActive] = useState(false);
  const [mobileVisionTab, setMobileVisionTab] = useState('radar'); // 'radar' | 'alerts'
  const [isVisionConnected, setIsVisionConnected] = useState(false);
  const [isHearingConnected, setIsHearingConnected] = useState(false);
  const [isInfoOpen, setIsInfoOpen] = useState(false);

  const handleDetectionUpdate = useCallback((data) => {
    setVisionData(data);
  }, []);

  useEffect(() => {
    if (!isInfoOpen) return;
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setIsInfoOpen(false);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [isInfoOpen]);

  const activeConnected = activeTab === 'vision' ? isVisionConnected : isHearingConnected;

  const tabs = [
    { id: 'vision', icon: '👁️', label: 'Vision Guide', subtitle: 'Spatial AI & Sonar' },
    { id: 'hearing', icon: '🎙️', label: 'Hearing AI', subtitle: 'Multimodal Audio' },
  ];

  return (
    <div
      className="relative flex flex-col md:flex-row h-[100dvh] bg-[#09090b] text-zinc-100 overflow-hidden font-sans selection:bg-zinc-800 selection:text-white"
      role="application"
      aria-label="AI Accessibility Assistant"
    >
      {/* Sidebar — desktop */}
      <aside
        className="hidden md:flex w-[72px] flex-shrink-0 bg-[#0c0c0e] border-r border-zinc-800/80 flex-col items-center py-5 z-20"
        aria-label="Main navigation"
      >
        <div className="relative mb-6 pb-4 border-b border-zinc-800/80 w-full flex items-center justify-center">
          <div
            className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-700/60 flex items-center justify-center text-zinc-100 shadow-sm"
            aria-hidden="true"
          >
            <span className="text-lg">♿</span>
          </div>
        </div>

        <nav className="flex flex-col gap-2.5 flex-1 w-full px-2.5" role="tablist" aria-label="Mode selection">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                role="tab"
                aria-selected={isActive}
                aria-label={`${tab.label} — ${tab.subtitle}`}
                className={`relative group w-full aspect-square rounded-xl flex flex-col items-center justify-center transition-all cursor-pointer ${
                  isActive
                    ? 'bg-zinc-800 text-white border border-zinc-700/60 shadow-sm'
                    : 'text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900/60 border border-transparent'
                }`}
                title={`${tab.label} — ${tab.subtitle}`}
              >
                <span className="text-lg">{tab.icon}</span>
                <span className="text-[0.55rem] font-semibold mt-1 tracking-tight">{tab.label.split(' ')[0]}</span>
                {isActive && (
                  <div className="absolute -left-2.5 top-1/2 -translate-y-1/2 w-1 h-5 rounded-r-full bg-white shadow-sm" />
                )}
              </button>
            );
          })}
        </nav>

        <button
          type="button"
          onClick={() => setIsInfoOpen(true)}
          className="w-9 h-9 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-xs text-zinc-400 hover:text-white hover:border-zinc-700 cursor-pointer transition-colors"
          title="Project Info & Model Architecture"
          aria-label="Open Project Info and Model Architecture"
        >
          ℹ️
        </button>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 z-10 overflow-hidden">
        {/* Top Header Bar */}
        <header className="bg-[#0c0c0e] border-b border-zinc-800/80 px-3.5 sm:px-4 md:px-6 py-2.5 sm:py-3 pt-safe flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2.5">
            <span className="text-xl md:hidden">♿</span>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-sm sm:text-base font-bold tracking-tight text-white">OMNI-ACCESS</h1>
                <span className="hidden sm:inline-block text-[0.62rem] font-medium px-2 py-0.5 rounded-full bg-zinc-900 text-zinc-400 border border-zinc-800 font-mono">
                  7th Sem Major Project
                </span>
              </div>
              <p className="hidden md:block text-[0.68rem] text-zinc-500 font-normal">
                {activeTab === 'vision'
                  ? 'Dual-Layer Navigation: YOLO11m Edge Reflexes + Gemini Multimodal Scene AI'
                  : 'Real-Time Conversational STT, Acoustic Event Radar & Cognitive Summarizer'}
              </p>
            </div>
          </div>

          {/* Desktop Mode Segmented Switcher */}
          <div className="hidden md:flex items-center p-1 rounded-xl bg-zinc-900/90 border border-zinc-800">
            {tabs.map((tab) => {
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                    isActive
                      ? 'bg-zinc-800 text-white border border-zinc-700/60 shadow-sm'
                      : 'text-zinc-400 hover:text-zinc-200'
                  }`}
                >
                  <span>{tab.icon}</span>
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Status Telemetry Badges */}
          <div className="flex items-center gap-2">
            <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-zinc-900 border border-zinc-800">
              <span className="text-[0.65rem] font-mono text-zinc-500">AI:</span>
              <span className="text-[0.65rem] font-mono text-zinc-300 font-semibold">
                {activeTab === 'vision' ? 'YOLO11 + GEMINI' : 'GEMINI AUDIO'}
              </span>
            </div>

            <div
              className={`flex items-center gap-1.5 px-2 sm:px-2.5 py-1 rounded-full border ${
                activeConnected
                  ? 'bg-emerald-500/10 border-emerald-500/20'
                  : 'bg-amber-500/10 border-amber-500/20'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  activeConnected ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'
                }`}
              />
              <span
                className={`text-[0.62rem] font-bold font-mono tracking-wider ${
                  activeConnected ? 'text-emerald-400' : 'text-amber-300'
                }`}
              >
                {activeConnected ? 'LIVE' : 'OFFLINE'}
              </span>
            </div>

            <button
              type="button"
              onClick={() => setIsInfoOpen(true)}
              className="md:hidden w-8 h-8 rounded-lg bg-zinc-900 border border-zinc-800 flex items-center justify-center text-xs text-zinc-400 hover:text-white cursor-pointer"
              aria-label="Open Project Info and Model Architecture"
            >
              ℹ️
            </button>
          </div>
        </header>

        {/* Dashboard Content Grid */}
        <main className="flex-1 flex flex-col p-2 sm:p-3 md:p-4 min-h-0 overflow-y-auto lg:overflow-hidden pb-20 md:pb-4">
          {/* Vision View Grid */}
          <div
            role="tabpanel"
            aria-label="Vision Guide Dashboard"
            className={`flex-1 flex flex-col lg:grid lg:grid-cols-[1.35fr_1fr] lg:grid-rows-[1.45fr_1fr] gap-2.5 md:gap-3 h-full ${activeTab === 'vision' ? '' : 'hidden'}`}
          >
            {/* Primary Camera Feed Viewport */}
            <div
              className={
                isGuidanceActive
                  ? 'fixed inset-0 z-50 bg-black flex flex-col w-screen h-[100dvh] overflow-hidden m-0 p-0'
                  : 'lg:row-span-2 glass-card rounded-2xl overflow-hidden flex flex-col min-h-[280px] sm:min-h-[320px] lg:min-h-0 shadow-lg relative'
              }
            >
              <CameraFeed
                onDetectionUpdate={handleDetectionUpdate}
                onConnectionChange={setIsVisionConnected}
                isActive={activeTab === 'vision'}
                isGuidanceActive={isGuidanceActive}
                onToggleGuidance={(active) => setIsGuidanceActive(active)}
              />
            </div>

            {/* Mobile Sub-Tab Switcher */}
            <div
              role="tablist"
              aria-label="Vision telemetry panels"
              className="lg:hidden flex items-center justify-center p-1 bg-zinc-900/90 border border-zinc-800 rounded-xl my-0.5 flex-shrink-0"
            >
              <button
                type="button"
                role="tab"
                aria-selected={mobileVisionTab === 'radar'}
                onClick={() => setMobileVisionTab('radar')}
                className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
                  mobileVisionTab === 'radar'
                    ? 'bg-zinc-800 text-white border border-zinc-700/60 shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <span>📡</span>
                <span>Spatial Radar</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mobileVisionTab === 'alerts'}
                onClick={() => setMobileVisionTab('alerts')}
                className={`flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
                  mobileVisionTab === 'alerts'
                    ? 'bg-zinc-800 text-white border border-zinc-700/60 shadow-sm'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <span>🔔</span>
                <span>Live Telemetry</span>
              </button>
            </div>

            {/* Spatial Proximity Radar Panel */}
            <div
              className={`glass-card rounded-2xl overflow-hidden flex flex-col min-h-[250px] lg:min-h-0 shadow-md ${
                mobileVisionTab === 'radar' ? 'flex' : 'hidden lg:flex'
              }`}
            >
              <SpatialRadarPanel visionData={visionData} />
            </div>

            {/* Real-Time Event & Safety Telemetry Panel */}
            <div
              className={`glass-card rounded-2xl overflow-hidden flex flex-col min-h-[220px] lg:min-h-0 shadow-md ${
                mobileVisionTab === 'alerts' ? 'flex' : 'hidden lg:flex'
              }`}
            >
              <AlertPanel visionData={visionData} />
            </div>
          </div>

          {/* Hearing View Card */}
          <div
            role="tabpanel"
            aria-label="Hearing and Speech Assistant"
            className={`flex-1 glass-card rounded-2xl overflow-hidden flex flex-col h-full shadow-md ${activeTab === 'hearing' ? '' : 'hidden'}`}
          >
            <HearingAssistant onConnectionChange={setIsHearingConnected} />
          </div>
        </main>
      </div>

      {/* Mobile Bottom Navigation Bar */}
      {!isGuidanceActive && (
        <nav
          className="md:hidden fixed bottom-0 left-0 right-0 bg-[#0c0c0e]/95 border-t border-zinc-800/80 backdrop-blur-xl flex justify-around items-center pt-1.5 z-40 shadow-lg"
          style={{ paddingBottom: 'max(0.65rem, env(safe-area-inset-bottom, 0.65rem))' }}
          role="tablist"
          aria-label="Mode selection"
        >
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                role="tab"
                aria-selected={isActive}
                aria-label={`${tab.label} — ${tab.subtitle}`}
                className={`flex flex-col items-center gap-0.5 px-6 py-1.5 rounded-xl transition-all cursor-pointer active:scale-95 ${
                  isActive
                    ? 'bg-zinc-800 text-white border border-zinc-700/60 font-semibold'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
              >
                <span className="text-lg">{tab.icon}</span>
                <span className="text-[0.62rem] tracking-tight">{tab.label}</span>
              </button>
            );
          })}
        </nav>
      )}

      {/* Project Info & Architecture Modal */}
      {isInfoOpen && (
        <div
          className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-label="System Architecture and Controls"
          onClick={() => setIsInfoOpen(false)}
        >
          <div
            className="w-full max-w-md bg-[#0c0c0e] border border-zinc-800 rounded-2xl p-5 shadow-2xl space-y-4 animate-fadeIn"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-xl">♿</span>
                <h2 className="text-sm font-bold text-white tracking-tight">OMNI-ACCESS Architecture</h2>
              </div>
              <button
                type="button"
                onClick={() => setIsInfoOpen(false)}
                className="px-2.5 py-1 rounded-lg bg-zinc-900 hover:bg-zinc-800 border border-zinc-800 text-xs text-zinc-300 cursor-pointer"
              >
                ✕ Close
              </button>
            </div>

            <div className="space-y-2.5 text-xs text-zinc-300 leading-relaxed">
              <div className="p-2.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
                <p className="font-bold text-zinc-100 mb-0.5">👁️ Vision Navigation Stack</p>
                <p className="text-zinc-400">
                  YOLO11m + isolated ByteTrack tracking + monocular metric distance + DeepFace emotion recognition + Gemini Vision cues + EasyOCR text reading.
                </p>
              </div>
              <div className="p-2.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80">
                <p className="font-bold text-zinc-100 mb-0.5">🎙️ Hearing & Speech Stack</p>
                <p className="text-zinc-400">
                  16kHz downsampled WAV capture → Gemini Multimodal Audio (with local Faster-Whisper fallback) + tone/urgency detection + Web Speech reply synthesis.
                </p>
              </div>
              <p className="text-[0.68rem] text-zinc-500 font-mono">
                Shortcut: Press Esc to exit Fullscreen Guidance or close dialogs.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
