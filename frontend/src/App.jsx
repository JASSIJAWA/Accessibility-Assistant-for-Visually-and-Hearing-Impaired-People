import { useState, useCallback } from 'react'
import CameraFeed from './components/CameraFeed'
import SpatialRadarPanel from './components/SpatialRadarPanel'
import AlertPanel from './components/AlertPanel'
import HearingAssistant from './components/HearingAssistant'

function App() {
  const [visionData, setVisionData] = useState(null);
  const [activeTab, setActiveTab] = useState('vision'); // 'vision' | 'hearing'
  const [isGuidanceActive, setIsGuidanceActive] = useState(false);

  const handleDetectionUpdate = useCallback((data) => {
    setVisionData(data);
  }, []);

  const tabs = [
    { id: 'vision',  icon: '👁️',  label: 'Vision Guide',  subtitle: 'Spatial AI & Sonar' },
    { id: 'hearing', icon: '🎙️', label: 'Hearing AI',    subtitle: 'Multimodal Audio' },
  ];

  return (
    <div className="relative flex flex-col md:flex-row h-screen bg-[#070714] text-gray-100 overflow-hidden font-sans selection:bg-purple-500/30 selection:text-white">
      
      {/* Background Ambient Radial Glows */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-purple-900/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-indigo-900/15 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-950/10 rounded-full blur-[140px] pointer-events-none" />

      {/* Sidebar — desktop (Hidden in Fullscreen Guidance) */}
      <aside className={`${isGuidanceActive ? 'hidden' : 'hidden md:flex'} w-[76px] flex-shrink-0 glass-panel border-r border-white/[0.08] flex-col items-center py-5 z-20`}>
        {/* Brand Icon */}
        <div className="relative mb-6 pb-4 border-b border-white/[0.08] w-full flex items-center justify-center">
          <div className="w-11 h-11 rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-500 border border-purple-400/50 flex items-center justify-center shadow-[0_0_20px_rgba(168,85,247,0.4)]">
            <span className="text-xl">♿</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex flex-col gap-3 flex-1 w-full px-2.5">
          {tabs.map(tab => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative group w-full aspect-square rounded-2xl flex flex-col items-center justify-center transition-all duration-300 cursor-pointer ${
                  isActive
                    ? 'bg-gradient-to-b from-purple-500/25 to-indigo-500/20 border border-purple-400/50 shadow-[0_0_15px_rgba(168,85,247,0.3)] text-white'
                    : 'text-white/40 hover:text-white/80 hover:bg-white/[0.05] border border-transparent'
                }`}
                title={`${tab.label} — ${tab.subtitle}`}
              >
                <span className="text-xl group-hover:scale-110 transition-transform">{tab.icon}</span>
                <span className="text-[0.55rem] font-bold mt-1 tracking-tight">{tab.label.split(' ')[0]}</span>
                {isActive && (
                  <div className="absolute -left-2.5 top-1/2 -translate-y-1/2 w-1.5 h-6 rounded-r-full bg-purple-400 shadow-[0_0_8px_rgba(168,85,247,0.9)]" />
                )}
              </button>
            );
          })}
        </nav>

        {/* Quick Help / Info Pill */}
        <div className="w-10 h-10 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-sm text-white/50 hover:text-white cursor-pointer transition-colors" title="Project Info & Model Architecture">
          ℹ️
        </div>
      </aside>

      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col min-w-0 z-10 ${isGuidanceActive ? 'h-full w-full' : 'overflow-hidden'}`}>

        {/* Top Header Bar (Hidden in Fullscreen Guidance) */}
        {!isGuidanceActive && (
          <header className="glass-panel border-b border-white/[0.08] px-4 md:px-6 py-2.5 md:py-3.5 flex items-center justify-between flex-shrink-0">
            {/* Title & Brand */}
            <div className="flex items-center gap-3">
              <span className="text-2xl md:hidden">♿</span>
              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-base md:text-lg font-black tracking-tight bg-gradient-to-r from-purple-300 via-indigo-200 to-cyan-300 bg-clip-text text-transparent">
                    OMNI-ACCESS AI
                  </h1>
                  <span className="hidden sm:inline-block text-[0.62rem] font-bold px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30 uppercase tracking-widest font-mono">
                    7th Sem Major Project
                  </span>
                </div>
                <p className="hidden md:block text-[0.7rem] text-white/40 font-medium">
                  {activeTab === 'vision'
                    ? 'Dual-Layer Navigation: YOLO11m Edge Reflexes + Gemini Multimodal Scene AI'
                    : 'Real-Time Conversational STT, Acoustic Event Radar & Cognitive Summarizer'
                  }
                </p>
              </div>
            </div>

            {/* Desktop Mode Segmented Switcher */}
            <div className="hidden md:flex items-center p-1 rounded-xl bg-[#0b0b1c]/80 border border-white/[0.08] shadow-inner">
              {tabs.map(tab => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-bold transition-all cursor-pointer ${
                      isActive
                        ? 'bg-gradient-to-r from-purple-600/80 to-indigo-600/80 text-white shadow-[0_0_12px_rgba(168,85,247,0.4)] border border-purple-400/40'
                        : 'text-white/50 hover:text-white/80'
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
              <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-white/[0.03] border border-white/[0.07]">
                <span className="text-[0.65rem] font-mono text-white/50 font-semibold">ENGINE:</span>
                <span className="text-[0.65rem] font-mono text-purple-300 font-bold">
                  {activeTab === 'vision' ? 'YOLO11 + GEMINI' : 'GEMINI AUDIO'}
                </span>
              </div>

              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                <span className="text-[0.65rem] font-bold text-emerald-400 font-mono tracking-wider">LIVE</span>
              </div>
            </div>
          </header>
        )}

        {/* Dashboard Content Grid */}
        <main className={`flex-1 flex flex-col ${isGuidanceActive ? 'p-0 m-0 h-[100dvh] w-full overflow-hidden' : 'p-3 md:p-4 min-h-0 overflow-y-auto lg:overflow-hidden pb-20 md:pb-4'}`}>

          {/* Vision View Grid */}
          <div className={`flex-1 flex flex-col ${isGuidanceActive ? 'w-full h-full' : 'lg:grid lg:grid-cols-[1.35fr_1fr] lg:grid-rows-[1.45fr_1fr] gap-3 h-full'} ${activeTab === 'vision' ? '' : 'hidden'}`}>
            
            {/* Primary Camera Feed Viewport */}
            <div className={isGuidanceActive
              ? 'w-full h-full flex flex-col bg-black overflow-hidden'
              : 'lg:row-span-2 glass-card rounded-2xl overflow-hidden flex flex-col min-h-[300px] lg:min-h-0 shadow-2xl relative'
            }>
              <CameraFeed
                onDetectionUpdate={handleDetectionUpdate}
                isActive={activeTab === 'vision'}
                isGuidanceActive={isGuidanceActive}
                onToggleGuidance={(active) => setIsGuidanceActive(active)}
              />
            </div>

            {/* Top Right: 2D Spatial Proximity Radar */}
            {!isGuidanceActive && (
              <div className="glass-card rounded-2xl overflow-hidden flex flex-col min-h-[260px] lg:min-h-0 shadow-xl">
                <SpatialRadarPanel visionData={visionData} />
              </div>
            )}

            {/* Bottom Right: Real-Time Event & Safety Telemetry */}
            {!isGuidanceActive && (
              <div className="glass-card rounded-2xl overflow-hidden flex flex-col min-h-[220px] lg:min-h-0 shadow-xl">
                <AlertPanel visionData={visionData} />
              </div>
            )}
          </div>

          {/* Hearing View Card */}
          <div className={`flex-1 glass-card rounded-2xl overflow-hidden flex flex-col h-full min-h-[400px] shadow-2xl ${activeTab === 'hearing' && !isGuidanceActive ? '' : 'hidden'}`}>
            <HearingAssistant />
          </div>

        </main>
      </div>

      {/* Mobile Glassmorphic Bottom Navigation Bar */}
      {!isGuidanceActive && (
        <nav className="md:hidden fixed bottom-0 left-0 right-0 glass-panel border-t border-white/[0.1] flex justify-around items-center py-2 px-4 z-50 shadow-[0_-5px_25px_rgba(0,0,0,0.5)]">
          {tabs.map(tab => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex flex-col items-center gap-0.5 px-4 py-1.5 rounded-xl transition-all cursor-pointer ${
                  isActive
                    ? 'bg-purple-500/25 text-purple-200 border border-purple-400/40 shadow-[0_0_12px_rgba(168,85,247,0.3)]'
                    : 'text-white/40 hover:text-white/80'
                }`}
              >
                <span className="text-xl">{tab.icon}</span>
                <span className="text-[0.62rem] font-bold tracking-tight">{tab.label}</span>
              </button>
            );
          })}
        </nav>
      )}

    </div>
  )
}

export default App
