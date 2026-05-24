"""Comparison schemes for the Phase 1 sweep.

Each scheme defines how an incoming task prompt is mapped into a sequence
of model calls under cap L. Stubs — implemented in Phase 1.
"""

from __future__ import annotations

SCHEMES = ("paged", "summarized", "rag", "subagent", "native")
