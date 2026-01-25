"""Centralized rate limiting and retry logic for API calls."""

import time
import random
import sys
import traceback
from typing import Callable, TypeVar, Any

T = TypeVar("T")


def _log(msg):
    """Debug logging to stderr."""
    print(f"[RATE_LIMITER] {msg}", file=sys.stderr, flush=True)


class RateLimiter:
    """Simple rate limiter with exponential backoff retry."""

    def __init__(self, min_delay: float = 1.0, max_retries: int = 3):
        """
        Initialize rate limiter.

        Args:
            min_delay: Minimum delay between API calls in seconds
            max_retries: Maximum number of retry attempts on failure
        """
        self.min_delay = min_delay
        self.max_retries = max_retries
        self.last_call = 0.0

    def wait(self) -> None:
        """Enforce minimum delay between calls."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self.last_call = time.time()

    def call_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call function with exponential backoff on failure.

        Args:
            func: Function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of the function call

        Raises:
            Exception: Re-raises the last exception after all retries exhausted
        """
        last_exception = None
        _log(f"call_with_retry() starting, max_retries={self.max_retries}")

        for attempt in range(self.max_retries):
            try:
                _log(f"  Attempt {attempt + 1}/{self.max_retries}: waiting...")
                self.wait()
                _log(f"  Attempt {attempt + 1}/{self.max_retries}: calling function...")
                result = func(*args, **kwargs)
                _log(f"  Attempt {attempt + 1}/{self.max_retries}: SUCCESS")
                return result
            except Exception as e:
                last_exception = e
                _log(f"  Attempt {attempt + 1}/{self.max_retries}: FAILED - {type(e).__name__}: {e}")
                _log(f"  Full traceback:\n{traceback.format_exc()}")

                if attempt == self.max_retries - 1:
                    # Last attempt, re-raise
                    _log(f"  All {self.max_retries} attempts exhausted, re-raising exception")
                    raise

                # Exponential backoff with jitter
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                _log(f"  Retrying in {wait_time:.1f}s...")
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)

        # Should never reach here, but satisfy type checker
        raise last_exception  # type: ignore


# Global instances with tuned delays
# Replicate: 3000 req/min = 50 req/s, reduced delay for faster batch processing
image_limiter = RateLimiter(min_delay=0.5, max_retries=3)

# OpenAI TTS: more generous limits, shorter delay
openai_limiter = RateLimiter(min_delay=0.5, max_retries=3)
