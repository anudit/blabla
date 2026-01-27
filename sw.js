// sw.js
const cacheName = 'bla-bla-v1';
const appAssets = [
  '/',
  '/index.html',
  '/bundle.js',       // Changed from frontend.tsx
  '/logo.png',
  '/tts.worker.js',
  '/pdf.worker.mjs',
  '/manifest.json'
];

// Install: Cache core assets
self.addEventListener('install', (e) => {
  console.log('[SW] Installing...');
  // Force SW to activate immediately
  self.skipWaiting();

  e.waitUntil(
    caches.open(cacheName).then((cache) => {
      console.log('[SW] Caching all assets');
      return cache.addAll(appAssets);
    })
  );
});

// Activate: Clean up old caches if needed
self.addEventListener('activate', (e) => {
  console.log('[SW] Activated');
  // Take control of all open clients immediately
  e.waitUntil(self.clients.claim());
});

// Fetch: Network first (for dev) or Cache first (for prod)
// For this PWA to work offline reliably, we use Cache First, falling back to Network.
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Helper to handle navigation requests (SPA support)
  if (e.request.mode === 'navigate') {
    e.respondWith(
      caches.match('/index.html').then((r) => r || fetch(e.request))
    );
    return;
  }

  e.respondWith(
    caches.match(e.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }

      return fetch(e.request).then((networkResponse) => {
        // Don't cache data from other domains or invalid responses
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(cacheName).then((cache) => {
          cache.put(e.request, responseToCache);
        });

        return networkResponse;
      }).catch((err) => {
        console.log('[SW] Fetch failed (offline):', e.request.url);
      });
    })
  );
});
