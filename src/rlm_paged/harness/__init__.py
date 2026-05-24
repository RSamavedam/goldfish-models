"""Harness for goldfish-models.

Two architectures coexist in this package:

- **Stateless-turn (canonical)**: `stateless_runner.run_stateless_cell`,
  `turn.assemble_input`, `turn.process_response`. The model is treated
  as a stateless coroutine; state lives in the BlockStore. This is the
  architecture going forward (see DESIGN.md).

- **Legacy paged-active-window**: `runner.run_cell`, `schemes.PagedScheme`
  et al, `conversation.ConversationState`. Preserved as a baseline so
  the interleaved-thinking Anthropic path keeps working and we can
  ablate against the new architecture.
"""

# Canonical (stateless-turn) surface
from rlm_paged.harness.stateless_runner import (
    StatelessCell,
    StatelessResult,
    run_stateless_cell,
)
from rlm_paged.harness.turn import (
    TurnInput,
    TurnOutput,
    assemble_input,
    process_response,
)

# Shared infrastructure
from rlm_paged.harness.cost_cap import CostCap, CostCapExceeded

# Legacy surface (preserved for interleaved-thinking baseline)
from rlm_paged.harness.conversation import ConversationState, Segment
from rlm_paged.harness.runner import RunResult, SweepCell, run_cell
from rlm_paged.harness.schemes import (
    SCHEMES,
    NativeScheme,
    PagedScheme,
    Scheme,
    SchemeContext,
    SummarizedScheme,
    TruncatedScheme,
    build_scheme,
)

__all__ = [
    # canonical
    "StatelessCell",
    "StatelessResult",
    "TurnInput",
    "TurnOutput",
    "assemble_input",
    "process_response",
    "run_stateless_cell",
    # shared
    "CostCap",
    "CostCapExceeded",
    # legacy
    "ConversationState",
    "NativeScheme",
    "PagedScheme",
    "RunResult",
    "SCHEMES",
    "Scheme",
    "SchemeContext",
    "Segment",
    "SummarizedScheme",
    "SweepCell",
    "TruncatedScheme",
    "build_scheme",
    "run_cell",
]
