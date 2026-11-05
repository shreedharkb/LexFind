import os

# Test-only JWT secret so plain `pytest` passes the app startup validation
# (at least 32 characters, not a known placeholder) without weakening the
# production check itself.
os.environ.setdefault("JWT_SECRET_KEY", "lexfind-test-only-secret-0123456789abcdef")
