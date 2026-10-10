// Minimal service worker — cache the UI shell for offline use.
const CACHE = "dolphin-v1";
const SHELL = ["/", "/index.html", "/styles.css", "/app.js", "/manifest.json", "/favicon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))),
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  // Only handle GETs on same-origin.
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  // Never cache API calls — always go to network.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/v1/") || url.pathname.startsWith("/health")) {
    return;
  }
  event.respondWith(
    caches.match(req).then((cached) => {
      const fetchPromise = fetch(req).then((resp) => {
        if (resp && resp.ok && resp.status === 200) {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        }
        return resp;
      }).catch(() => cached || caches.match("/"));
      return cached || fetchPromise;
    }),
  );
});
