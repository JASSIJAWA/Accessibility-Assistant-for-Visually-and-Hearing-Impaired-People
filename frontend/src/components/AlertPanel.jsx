import { useState, useEffect } from 'react';

function AlertPanel({ visionData }) {
  const [alerts, setAlerts] = useState([
    { id: 1, type: 'info', icon: '🟢', message: 'Vision AI & Spatial Engine Active', time: 'Just now' },
  ]);

  const objects = visionData?.objects || [];
  const emotions = visionData?.emotions || [];
  const dominantEmotion = emotions[0]?.emotion || '--';
  const closestDist = visionData?.closest_distance_m;
  const isCritical = visionData?.is_critical_hazard;

  const emotionEmoji = {
    happy: '😊', sad: '😢', angry: '😠', surprise: '😲',
    fear: '😨', disgust: '🤢', neutral: '😐',
  };

  // Add real-time alerts when obstacles or faces enter frame
  useEffect(() => {
    if (!visionData) return;

    // 1. Critical Obstacle Warning Alert
    if (isCritical && closestDist) {
      const criticalObj = objects.find(o => o.urgency === 'critical') || objects[0];
      const newAlert = {
        id: Date.now() + 1,
        type: 'danger',
        icon: '⚠️',
        message: `Collision hazard: ${criticalObj?.label || 'Obstacle'} at ${closestDist}m ahead!`,
        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
      };

      setAlerts((prev) => {
        if (prev[0] && prev[0].message === newAlert.message) return prev;
        return [newAlert, ...prev].slice(0, 20);
      });
    }

    // 2. Emotion alert
    if (emotions.length > 0) {
      const emotion = emotions[0];
      if (emotion.confidence > 60) {
        const newAlert = {
          id: Date.now(),
          type: emotion.emotion === 'angry' ? 'danger' : emotion.emotion === 'happy' ? 'info' : 'warning',
          icon: emotionEmoji[emotion.emotion] || '👤',
          message: `Person detected (${emotion.emotion}, ${Math.round(emotion.confidence)}% conf)`,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        };

        setAlerts((prev) => {
          if (prev[0] && prev[0].message === newAlert.message) return prev;
          return [newAlert, ...prev].slice(0, 20);
        });
      }
    }
  }, [visionData, isCritical, closestDist, emotions, objects]);

  const stats = [
    {
      icon: '👁️',
      value: String(objects.length),
      label: 'Tracked',
      highlight: objects.length > 0 ? 'text-purple-300' : 'text-gray-400'
    },
    {
      icon: '📏',
      value: closestDist ? `${closestDist}m` : 'Safe',
      label: 'Closest',
      highlight: isCritical ? 'text-red-400 font-bold' : closestDist ? 'text-amber-300' : 'text-emerald-400'
    },
    {
      icon: emotionEmoji[dominantEmotion] || '👤',
      value: dominantEmotion,
      label: 'Emotion',
      highlight: 'text-indigo-300'
    },
    {
      icon: '📡',
      value: isCritical ? 'Critical' : 'Normal',
      label: 'Reflex',
      highlight: isCritical ? 'text-red-400' : 'text-emerald-400'
    },
  ];

  const borderColors = {
    info: 'border-l-emerald-400 bg-emerald-500/5',
    warning: 'border-l-amber-400 bg-amber-500/5',
    danger: 'border-l-red-500 bg-red-500/10 shadow-[0_0_12px_rgba(239,68,68,0.15)]',
  };

  return (
    <div className="flex flex-col h-full overflow-hidden bg-transparent">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.08] flex-shrink-0 bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <span className="text-base">🔔</span>
          <h2 className="text-xs md:text-sm font-bold text-gray-100 tracking-tight">Real-Time Event Telemetry</h2>
        </div>
        <button
          onClick={() => setAlerts([{ id: Date.now(), type: 'info', icon: '🟢', message: 'Event log cleared', time: 'Just now' }])}
          className="text-[0.6rem] text-white/40 hover:text-white/80 transition-colors uppercase font-mono tracking-wider cursor-pointer"
        >
          Clear Log
        </button>
      </div>

      {/* Stats Quick Cards */}
      <div className="grid grid-cols-4 gap-1.5 p-3 flex-shrink-0">
        {stats.map((stat) => (
          <div key={stat.label} className="flex flex-col items-center justify-center bg-white/[0.03] border border-white/[0.06] rounded-xl p-2 text-center">
            <span className="text-base mb-0.5">{stat.icon}</span>
            <p className={`text-xs font-bold font-mono leading-tight capitalize truncate w-full ${stat.highlight}`}>
              {stat.value}
            </p>
            <p className="text-white/30 text-[0.55rem] uppercase tracking-wider mt-0.5">{stat.label}</p>
          </div>
        ))}
      </div>

      {/* Event Stream Log */}
      <div className="flex-1 px-3 pb-3 overflow-y-auto min-h-0 flex flex-col gap-1.5">
        <div className="flex items-center justify-between px-1 mb-1">
          <h3 className="text-white/30 text-[0.62rem] uppercase tracking-widest font-bold font-mono">Live Activity Stream</h3>
          <span className="text-[0.6rem] text-emerald-400 font-mono">● Active</span>
        </div>

        {alerts.map((alert) => (
          <div
            key={alert.id}
            className={`flex items-start gap-2.5 p-2 rounded-xl border border-white/[0.05] border-l-[3px] transition-all ${borderColors[alert.type] || 'border-l-gray-500'}`}
          >
            <span className="text-sm flex-shrink-0 mt-0.5">{alert.icon}</span>
            <div className="flex-1 min-w-0">
              <p className="text-gray-100 text-xs font-medium leading-relaxed truncate">{alert.message}</p>
              <span className="text-white/30 text-[0.6rem] font-mono">{alert.time}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default AlertPanel;
