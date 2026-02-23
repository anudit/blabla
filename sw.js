// sw.js
const cacheName = 'bla-bla-v2'; // bumped to trigger old-cache cleanup
const appAssets = [
  '/',
  '/index.html',
  '/bundle.js',
  '/logo.png',
  '/tts.worker.js',
  '/pdf.worker.mjs',
  '/manifest.json'
];

// Install: cache core app assets
self.addEventListener('install', (e) => {
  console.log('[SW] Installing...');
  self.skipWaiting();
  e.waitUntil(
    caches.open(cacheName).then((cache) => {
      console.log('[SW] Caching core assets');
      return cache.addAll(appAssets);
    })
  );
});

// Activate: delete stale caches from previous versions
self.addEventListener('activate', (e) => {
  console.log('[SW] Activated');
  e.waitUntil(
    caches.keys().then((keyList) =>
      Promise.all(
        keyList.map((key) => {
          if (key !== cacheName) {
            console.log('[SW] Removing old cache:', key);
            return caches.delete(key);
          }
        })
      )
    ).then(() => self.clients.claim())
  );
});

// Fetch: cache-first for app assets, bypass for HuggingFace (IndexedDB handles those)
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // HuggingFace model files are cached in IndexedDB by the TTS worker — don't double-cache them
  if (url.hostname === 'huggingface.co') return;

  // SPA navigation → always serve index.html
  if (e.request.mode === 'navigate') {
    e.respondWith(caches.match('/index.html').then((r) => r || fetch(e.request)));
    return;
  }

  e.respondWith(
    caches.match(e.request).then((cachedResponse) => {
      if (cachedResponse) return cachedResponse;

      return fetch(e.request).then((networkResponse) => {
        // Only cache same-origin, successful, GET responses
        if (
          !networkResponse ||
          networkResponse.status !== 200 ||
          networkResponse.type !== 'basic' ||
          e.request.method !== 'GET'
        ) {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(cacheName).then((cache) => cache.put(e.request, responseToCache));
        return networkResponse;
      }).catch(() => {
        console.log('[SW] Fetch failed (offline):', e.request.url);
      });
    })
  );
});
