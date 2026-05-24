"""Op surfaces for goldfish-models.

Two op surfaces coexist (corresponding to the two architectures):

- **New (canonical)**: `note` / `continue` / `query` / `pipe` / `call`
  for the stateless-turn architecture. See `ops.py`, `executor.py`,
  `schema.render_system_prompt`.

- **Legacy**: single-char codes `e` / `r` / `q` / `a` / `l` / `s` for
  the deprecated paged-active-window design and the interleaved-thinking
  baseline. See `api.py`, `structured.py`, `schema.LEGACY_PREFIX_SCHEMA`.

Both surfaces are exported so existing tests and the preserved
interleaved-thinking baseline keep working unmodified.
"""

# New surface (canonical)
from rlm_paged.tools.executor import (
    ExecutionResult,
    ExternalCall,
    QueuedQuery,
    execute,
)
from rlm_paged.tools.ops import (
    OP_NAMES,
    Op,
    SCRATCH_CLOSE,
    SCRATCH_OPEN,
    extract_scratch,
    parse_ops,
)
from rlm_paged.tools.schema import (
    LEGACY_OP_REFERENCE,
    LEGACY_PREFIX_SCHEMA,
    OP_REFERENCE,
    PREFIX_SCHEMA,
    SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)

# Legacy surface (kept for the interleaved-thinking baseline)
from rlm_paged.tools.api import OpResult, ParsedOp, ToolDispatcher, parse_op
from rlm_paged.tools.structured import (
    ANTHROPIC_TOOLS,
    opcode_of,
    structured_to_parsed_op,
)

__all__ = [
    # new surface
    "OP_NAMES",
    "Op",
    "ExecutionResult",
    "ExternalCall",
    "QueuedQuery",
    "SCRATCH_CLOSE",
    "SCRATCH_OPEN",
    "SYSTEM_PROMPT_TEMPLATE",
    "execute",
    "extract_scratch",
    "parse_ops",
    "render_system_prompt",
    # shared / both
    "OP_REFERENCE",
    "PREFIX_SCHEMA",
    # legacy surface
    "ANTHROPIC_TOOLS",
    "LEGACY_OP_REFERENCE",
    "LEGACY_PREFIX_SCHEMA",
    "OpResult",
    "ParsedOp",
    "ToolDispatcher",
    "opcode_of",
    "parse_op",
    "structured_to_parsed_op",
]
