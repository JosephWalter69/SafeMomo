// service-worker.js – very basic for PWA caching
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open('momo-guard-v1').then((cache) => {
      return cache.addAll([
        '/',
        '/manifest.json',
        '/static/icon-192.png',
        '/static/icon-512.png'
      ]);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
