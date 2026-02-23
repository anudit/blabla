// sw.js
const CACHE_VERSION = 'bla-bla-v3';
const APP_SHELL = [
  '/',
  '/index.html',
  '/bundle.js',
  '/logo.png',
  '/tts.worker.js',
  '/pdf.worker.mjs',
  '/manifest.json'
];

// Install: pre-cache app shell and activate immediately
self.addEventListener('install', (e) => {
  console.log('[SW] Installing...');
  self.skipWaiting();
  e.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => {
      console.log('[SW] Caching app shell');
      return cache.addAll(APP_SHELL);
    })
  );
});

// Activate: clean up old bla-bla-vX caches only — leave transformers.js model caches alone
self.addEventListener('activate', (e) => {
  console.log('[SW] Activating...');
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.map((key) => {
          if (key.startsWith('bla-bla-') && key !== CACHE_VERSION) {
            console.log('[SW] Removing old cache:', key);
            return caches.delete(key);
          }
        })
      ))
      .then(() => self.clients.claim())
  );
});

// Fetch: network-first for app shell (picks up new deployments automatically),
// pass-through for cross-origin (model files handled by transformers.js Cache Storage)
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Only handle same-origin GET requests — let everything else pass through
  if (e.request.method !== 'GET' || url.origin !== self.location.origin) return;

  // SPA navigation → network-first, fall back to cached index.html when offline
  if (e.request.mode === 'navigate') {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(e.request, copy));
          return res;
        })
        .catch(() => caches.match('/index.html'))
    );
    return;
  }

  // App shell assets: network-first so new deployments are always fetched,
  // fall back to stale cache when offline
  if (APP_SHELL.includes(url.pathname)) {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          if (res.ok) {
            const copy = res.clone();
            caches.open(CACHE_VERSION).then((c) => c.put(e.request, copy));
          }
          return res;
        })
        .catch(() => {
          console.log('[SW] Offline, serving cached:', url.pathname);
          return caches.match(e.request);
        })
    );
    return;
  }

  // Other same-origin assets: cache-first
  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request).then((res) => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(e.request, copy));
        }
        return res;
      }).catch(() => {
        console.log('[SW] Fetch failed (offline):', url.pathname);
      });
    })
  );
});
