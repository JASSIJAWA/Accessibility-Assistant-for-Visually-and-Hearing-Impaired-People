// Backend endpoint resolution: uses env overrides in production deployments,
// or automatically resolves against the current origin (Vite dev proxy) locally.

const stripTrailingSlash = (url = '') => url.replace(/\/+$/, '');

export const getWsBaseUrl = () => {
  const configured = import.meta.env.VITE_WS_URL;
  if (configured) {
    return stripTrailingSlash(configured);
  }
  const apiUrl = import.meta.env.VITE_API_URL;
  if (apiUrl) {
    const clean = stripTrailingSlash(apiUrl);
    if (clean.startsWith('https://')) return clean.replace(/^https:\/\//, 'wss://');
    if (clean.startsWith('http://')) return clean.replace(/^http:\/\//, 'ws://');
    return clean;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}`;
};

export const getApiBaseUrl = () => {
  const configured = import.meta.env.VITE_API_URL;
  return configured ? stripTrailingSlash(configured) : '';
};

export const getVisionWsUrl = () => `${getWsBaseUrl()}/ws/vision`;
export const getHearingWsUrl = () => `${getWsBaseUrl()}/ws/hearing`;
export const getOcrApiUrl = () => `${getApiBaseUrl()}/api/ocr`;
