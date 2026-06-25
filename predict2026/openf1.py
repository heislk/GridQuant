"""
OpenF1 API client with on-disk caching.

OpenF1 is an unofficial project and is not associated with the Formula 1
companies. F1, FORMULA 1 and related marks are trade marks of Formula One
Licensing B.V. This module only consumes the public OpenF1 REST endpoints.
"""
import os
import json
import time
import hashlib
import urllib.parse
import urllib.request
import urllib.error

BASE_URL = "https://api.openf1.org/v1"


class OpenF1Client:
    """Thin caching wrapper over the OpenF1 REST API.

    Every GET is cached to ``cache_dir`` keyed by the full URL, so repeated
    pipeline runs (and the as-of historical pulls, which never change) hit the
    network only once.
    """

    def __init__(self, cache_dir, timeout=60, max_retries=4, pause=0.4):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.pause = pause
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, url):
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{h}.json")

    def get(self, endpoint, use_cache=True, **params):
        """GET ``/v1/{endpoint}`` with query params; returns parsed JSON list."""
        query = urllib.parse.urlencode(params, doseq=True)
        url = f"{BASE_URL}/{endpoint}"
        if query:
            url = f"{url}?{query}"

        path = self._cache_path(url)
        if use_cache and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

        data = self._fetch(url)
        with open(path, "w") as f:
            json.dump(data, f)
        return data

    def _fetch(self, url):
        last_err = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "GridQuant/2026 (+research)"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                return json.loads(raw)
            except urllib.error.HTTPError as e:
                # OpenF1 returns 404 for sessions with no published data yet
                # (a couple of 2026 races lack detailed results). Treat as empty.
                if e.code in (404, 422):
                    return []
                last_err = e
                time.sleep(self.pause * (2 ** attempt))
            except Exception as e:  # network / 429 / transient
                last_err = e
                time.sleep(self.pause * (2 ** attempt))
        raise RuntimeError(f"OpenF1 request failed after retries: {url}\n{last_err}")
