"""
In-memory rate limiter for LexFind API endpoints.
Falls back gracefully if Redis is not available.

Limits:
  - Chat (send_message): 30 req / minute per user
  - Search: 60 req / minute per user
"""
import time
import logging
from collections import defaultdict, deque
from threading import Lock

logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Thread-safe sliding window rate limiter backed by in-memory deques."""

    def __init__(self):
        self._windows: dict = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        """Return True if the request should be allowed, False if rate limited."""
        now = time.time()
        cutoff = now - window_seconds

        with self._lock:
            dq = self._windows[key]
            # Evict timestamps outside the window
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= max_requests:
                logger.warning("Rate limit exceeded for key=%s (%d/%d in %ds)", key, len(dq), max_requests, window_seconds)
                return False

            dq.append(now)
            return True

    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        """Return how many requests remain in the current window."""
        now    = time.time()
        cutoff = now - window_seconds
        with self._lock:
            dq = self._windows[key]
            count = sum(1 for t in dq if t > cutoff)
            return max(0, max_requests - count)


# Singleton
rate_limiter = InMemoryRateLimiter()


# ── Convenience wrappers ───────────────────────────────────────────────────────

def check_chat_rate(user_id: str) -> bool:
    """30 messages per minute per user."""
    return rate_limiter.is_allowed(f"chat:{user_id}", max_requests=30, window_seconds=60)


def check_search_rate(user_id: str) -> bool:
    """60 searches per minute per user."""
    return rate_limiter.is_allowed(f"search:{user_id}", max_requests=60, window_seconds=60)
