import os
import logging
from groq import Groq, AsyncGroq

logger = logging.getLogger(__name__)

_groq_client = None
_async_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        if api_key:
            _groq_client = Groq(api_key=api_key)
        else:
            logger.warning("GROQ_API_KEY is missing")
    return _groq_client

def get_async_groq_client() -> AsyncGroq:
    global _async_groq_client
    if _async_groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'")
        if api_key:
            _async_groq_client = AsyncGroq(api_key=api_key)
        else:
            logger.warning("GROQ_API_KEY is missing")
    return _async_groq_client
