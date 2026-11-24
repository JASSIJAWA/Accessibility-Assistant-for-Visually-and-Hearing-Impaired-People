import { useState } from 'react';

/**
 * SpatialRadarPanel — 2D Spatial Proximity Radar & Real-Time Navigation Telemetry.
 * Plotted in real-time from YOLO11 bounding box geometry and distance metrics.
 */
function SpatialRadarPanel({ visionData }) {
  const [isReplayingSpeech, setIsReplayingSpeech] = useState(false);

  const objects = visionData?.objects || [];
  const sceneDescription = visionData?.scene_description || visionData?.narration || '';
  const isCritical = visionData?.is_critical_hazard || false;
  const closestDist = visionData?.closest_distance_m;

  // Replay speech helper
  const handleReplaySpeech = () => {
    if (!sceneDescription || typeof window === 'undefined' || !('speechSynthesis' in window)) return;
    setIsReplayingSpeech(true);
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(sceneDescription);
    utterance.rate = 1.15;
    utterance.onend = () => setIsReplayingSpeech(false);
    utterance.onerror = () => setIsReplayingSpeech(false);
    window.speechSynthesis.speak(utterance);
  };

  // Convert object position ('on your left', 'ahead', 'on your right') and distance into (x, y) % on radar
  // Radar coordinate space: User at bottom-center (50%, 92%)
  const getRadarCoordinates = (obj) => {
    const dist = Math.min(Math.max(obj.distance_m || 2.0, 0.4), 3.5); // clamp 0.4m to 3.5m
    const normalizedDist = (dist - 0.3) / 3.2; // 0 (close) to 1 (far)
    const radiusPct = 15 + normalizedDist * 65; // % distance from user point

    // Angle offset: left = -35deg, ahead = 0deg, right = +35deg
    let angleDeg = 0;
    if (obj.position?.includes('left')) angleDeg = -38;
    else if (obj.position?.includes('right')) angleDeg = 38;

    // Convert polar (radius, angle from top 0) to cartesian %
    // User is at (50, 90)
    const angleRad = (angleDeg - 90) * (Math.PI / 180);
    const x = 50 + radiusPct * Math.cos(angleRad) * 0.95;
    const y = 90 + radiusPct * Math.sin(angleRad) * 0.95;

    return { x: Math.min(Math.max(x, 8), 92), y: Math.min(Math.max(y, 10), 88) };
  };

  return (
    <div className="flex flex-col h-full overflow-hidden bg-transparent">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/80 flex-shrink-0 bg-zinc-900/40">
        <div className="flex items-center gap-2">
          <span className="text-base">📡</span>
          <div>
            <h2 className="text-xs md:text-sm font-bold text-zinc-100 tracking-tight flex items-center gap-2">
              Spatial Proximity Radar
              <span className="text-[0.6rem] px-2 py-0.5 rounded-full bg-zinc-900 text-zinc-400 border border-zinc-800 font-mono">
                360° FOV
              </span>
            </h2>
          </div>
        </div>

        {/* Live Safety Indicator */}
        <div className="flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${
            isCritical
              ? 'bg-red-500 animate-ping'
              : objects.length > 0
                ? 'bg-amber-400'
                : 'bg-emerald-400'
          }`} />
          <span className={`text-[0.65rem] font-bold uppercase tracking-wider font-mono ${
            isCritical ? 'text-red-400' : objects.length > 0 ? 'text-amber-300' : 'text-emerald-400'
          }`}>
            {isCritical ? 'Hazard Alert' : objects.length > 0 ? 'Obstacle Near' : 'Path Clear'}
          </span>
        </div>
      </div>

      {/* Main Content Body */}
      <div className="flex-1 flex flex-col p-3 gap-3 min-h-0 overflow-y-auto">
        
        {/* 2D Mini Radar Screen */}
        <div className="relative w-full aspect-[2/1.05] bg-[#0c0c0e] border border-zinc-800 rounded-xl overflow-hidden flex items-center justify-center shadow-inner">
          
          {/* Subtle Grid / Background Rings */}
          <div className="absolute inset-0 opacity-20 bg-[radial-gradient(rgba(255,255,255,0.15)_1px,transparent_1px)] [background-size:16px_16px]" />

          {/* Distance Arcs (1m, 2m, 3m) */}
          <div className="absolute bottom-0 w-[40%] aspect-square rounded-full border border-red-500/25 pointer-events-none translate-y-1/2" />
          <div className="absolute bottom-0 w-[70%] aspect-square rounded-full border border-amber-400/20 pointer-events-none translate-y-1/2" />
          <div className="absolute bottom-0 w-[95%] aspect-square rounded-full border border-zinc-700/30 pointer-events-none translate-y-1/2" />

          {/* Distance Labels */}
          <span className="absolute bottom-1.5 left-3 text-[0.55rem] font-mono text-zinc-500 uppercase">3.0m Zone</span>
          <span className="absolute bottom-1.5 right-3 text-[0.55rem] font-mono text-zinc-500 uppercase">1.5m Zone</span>

          {/* Sector Lines */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-full bg-gradient-to-t from-white/10 via-white/5 to-transparent pointer-events-none" />
          <div className="absolute bottom-0 left-1/2 w-px h-full bg-white/5 origin-bottom -rotate-35 pointer-events-none" />
          <div className="absolute bottom-0 left-1/2 w-px h-full bg-white/5 origin-bottom rotate-35 pointer-events-none" />

          {/* User Marker at Bottom Center */}
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex flex-col items-center z-10">
            <div className="relative w-3.5 h-3.5 rounded-full bg-white border border-zinc-300 flex items-center justify-center shadow-sm">
              <span className="text-[0.45rem] leading-none text-black">▲</span>
            </div>
            <span className="text-[0.55rem] font-semibold text-zinc-500 tracking-wider uppercase font-mono mt-0.5">YOU</span>
          </div>

          {/* Plotted Object Blips */}
          {objects.slice(0, 8).map((obj, idx) => {
            const { x, y } = getRadarCoordinates(obj);
            const isCrit = obj.urgency === 'critical';
            const isWarn = obj.urgency === 'warning';

            const blipColor = isCrit
              ? 'bg-red-500 text-red-100 border-red-400 shadow-sm'
              : isWarn
                ? 'bg-amber-400 text-black border-amber-300 shadow-sm'
                : 'bg-zinc-200 text-black border-white shadow-sm';

            return (
              <div
                key={idx}
                className="absolute -translate-x-1/2 -translate-y-1/2 z-20 group transition-all duration-300"
                style={{ left: `${x}%`, top: `${y}%` }}
              >
                {/* Ping blip dot */}
                <div className={`w-3 h-3 rounded-full border flex items-center justify-center ${blipColor} cursor-pointer`}>
                  <div className="w-1 h-1 rounded-full bg-white" />
                </div>

                {/* Micro Label Tag */}
                <div className="absolute left-1/2 -translate-x-1/2 bottom-3.5 pointer-events-none opacity-90 group-hover:opacity-100 transition-opacity">
                  <span className={`text-[0.58rem] font-semibold px-1.5 py-0.5 rounded shadow whitespace-nowrap backdrop-blur-md border ${
                    isCrit ? 'bg-red-950/90 text-red-300 border-red-500/50' : 'bg-zinc-900/90 text-zinc-200 border-zinc-700'
                  }`}>
                    {obj.label} ({obj.distance_m ? `${obj.distance_m}m` : `${obj.distance_steps}s`})
                  </span>
                </div>
              </div>
            );
          })}

          {objects.length === 0 && (
            <div className="text-center z-10 pointer-events-none">
              <span className="text-lg opacity-40">🟢</span>
              <p className="text-[0.68rem] text-zinc-400 font-medium">Path Clear • No obstacles in 3m range</p>
            </div>
          )}
        </div>

        {/* Gemini Scene Intelligence Cue Card */}
        <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-3 shadow-sm relative overflow-hidden flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5 font-mono">
              <span>🧠</span> Gemini Scene Guidance:
            </span>
            {sceneDescription && (
              <button
                onClick={handleReplaySpeech}
                disabled={isReplayingSpeech}
                className="text-[0.65rem] font-semibold text-zinc-300 hover:text-white bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 px-2 py-0.5 rounded-md transition-colors flex items-center gap-1 cursor-pointer disabled:opacity-50"
                title="Speak Guidance Again"
              >
                <span>{isReplayingSpeech ? '🔊 Speaking...' : '🔊 Replay'}</span>
              </button>
            )}
          </div>
          <p className="text-xs text-zinc-200 font-medium leading-relaxed italic">
            "{sceneDescription || 'Observing surroundings... Path is clear.'}"
          </p>
        </div>

        {/* Telemetry Micro-Cards */}
        <div className="grid grid-cols-2 gap-2 mt-auto">
          <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-2.5 flex items-center gap-2">
            <span className="text-base">📏</span>
            <div>
              <p className="text-[0.6rem] text-zinc-500 uppercase tracking-wider font-semibold font-mono">Closest Distance</p>
              <p className="text-sm font-bold font-mono text-zinc-100">
                {closestDist ? `${closestDist} m` : 'Clear (>3m)'}
              </p>
            </div>
          </div>

          <div className="bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-2.5 flex items-center gap-2">
            <span className="text-base">🎯</span>
            <div>
              <p className="text-[0.6rem] text-zinc-500 uppercase tracking-wider font-semibold font-mono">Tracked Objects</p>
              <p className="text-sm font-bold font-mono text-zinc-100">
                {objects.length} <span className="text-[0.65rem] font-normal text-zinc-500">active</span>
              </p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default SpatialRadarPanel;
