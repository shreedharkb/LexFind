import pytest

from app.services.auth_service import validate_jwt_secret


def test_jwt_secret_rejects_placeholder(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "your_secret_key_here")

    with pytest.raises(RuntimeError, match="non-placeholder"):
        validate_jwt_secret()


def test_jwt_secret_rejects_short_value(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "too-short")

    with pytest.raises(RuntimeError, match="at least 32"):
        validate_jwt_secret()


def test_jwt_secret_accepts_long_random_value(monkeypatch):
    secret = "a" * 64
    monkeypatch.setenv("JWT_SECRET_KEY", secret)

    assert validate_jwt_secret() == secret
