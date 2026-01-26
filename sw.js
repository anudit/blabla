const cacheName = 'bla-bla-v1';
const appAssets = [
  '/',
  '/index.html',
  '/frontend.tsx',
  '/logo.png',
  '/tts.worker.js',
  '/pdf.worker.mjs',
  '/manifest.json',
  // Add any other static assets (e.g., if you have CSS files)
];

// Install event: Cache assets
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(cacheName).then((cache) => {
      return cache.addAll(appAssets);
    })
  );
});

// Fetch event: Serve from cache, fallback to network, cache new resources
self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(e.request).then((networkResponse) => {
        return caches.open(cacheName).then((cache) => {
          cache.put(e.request, networkResponse.clone());
          return networkResponse;
        });
      }).catch(() => {
        // Offline fallback: Serve cached index.html for navigations
        if (e.request.mode === 'navigate') {
          return caches.match('/index.html');
        }
      });
    })
  );
});
