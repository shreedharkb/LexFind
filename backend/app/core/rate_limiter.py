"""
In-memory rate limiter for LexFind API endpoints.
Falls back gracefully if Redis is not available.

Limits:
  - Chat (send_message): 30 req / minute per user
  - Search: 60 req / minute per user
"""
import os
import time
import logging
from collections import defaultdict, deque
from threading import Lock
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


class RedisRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, now - window_seconds)
        pipeline.zcard(key)
        pipeline.zadd(key, {str(now): now})
        pipeline.expire(key, window_seconds)
        results = pipeline.execute()
        
        count = results[1]
        # In this implementation, since we zadd in the same pipeline before checking, 
        # the count we get is before we added the new one. So we check against max_requests
        if count >= max_requests:
            # We should probably remove the one we just added if it's over the limit
            self.redis.zrem(key, str(now))
            logger.warning("Rate limit exceeded for key=%s (%d/%d in %ds)", key, count, max_requests, window_seconds)
            return False
        return True

    def get_remaining(self, key: str, max_requests: int, window_seconds: int) -> int:
        now = time.time()
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(key, 0, now - window_seconds)
        pipeline.zcard(key)
        results = pipeline.execute()
        return max(0, max_requests - results[1])


def _get_rate_limiter():
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis
            client = redis.from_url(redis_url)
            client.ping()
            logger.info("Using RedisRateLimiter")
            return RedisRateLimiter(client)
        except Exception as e:
            logger.warning("Redis not available, falling back to in-memory: %s", e)
    
    return InMemoryRateLimiter()

# Singleton
rate_limiter = _get_rate_limiter()


# ── Convenience wrappers ───────────────────────────────────────────────────────

def check_chat_rate(user_id: str) -> bool:
    """30 messages per minute per user."""
    return rate_limiter.is_allowed(f"chat:{user_id}", max_requests=30, window_seconds=60)


def check_search_rate(user_id: str) -> bool:
    """60 searches per minute per user."""
    return rate_limiter.is_allowed(f"search:{user_id}", max_requests=60, window_seconds=60)
