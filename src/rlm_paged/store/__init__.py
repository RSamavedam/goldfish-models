"""Block / chunk storage for goldfish-models.

Two storage types coexist:

- `BlockStore` (in `block_store.py`): the typed, append-only block store
  used by the stateless-turn architecture. Operates on text. This is the
  canonical store going forward.

- `ChunkStore` (in `store.py`): the legacy token-ID chunk store used by
  the deprecated paged-active-window design and the interleaved-thinking
  baseline. Kept for backward compatibility while those paths are
  preserved as alternate scheme baselines.

See DESIGN.md for which architecture is which.
"""

from rlm_paged.store.block import BLOCK_TYPES, Block
from rlm_paged.store.block_store import BlockStore
from rlm_paged.store.chunk import Chunk
from rlm_paged.store.store import ChunkStore

__all__ = [
    "BLOCK_TYPES",
    "Block",
    "BlockStore",
    "Chunk",
    "ChunkStore",
]
