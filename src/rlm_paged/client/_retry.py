"""Tiny retry wrapper for provider API calls.

Exponential backoff with jitter, retries only on rate-limit / transient errors.
Each provider's SDK has its own retry, but those defaults are tuned for
production traffic; here we want longer waits because we're running batched
sweeps where one rate-limit can be the difference between $1 and $30.
"""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 1.5,
    max_delay: float = 60.0,
    retryable: tuple[type[BaseException], ...] = (),
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            delay *= 0.5 + random.random()  # 0.5-1.5x jitter
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
