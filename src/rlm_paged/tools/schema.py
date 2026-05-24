"""Token-frugal op-code schema for the model-facing tool API.

Lives in the pinned prefix. At L=32, JSON-style tool calls don't fit — so
ops are single ASCII characters and args are positional integers.

Wire format examples (one per line in the model's output stream):

    e 128            evict 128 tokens from the head
    r 42 0 64        retrieve 64 tokens from chunk 42 starting at offset 0
    q 42             return refs (outgoing,incoming) of chunk 42
    a 42 plan        tag chunk 42 with "plan"
    l 7 42           link chunk 7 -> chunk 42
    s how-to-deploy 3   top-3 similarity search (when embeddings on)
"""

from __future__ import annotations

OP_REFERENCE: dict[str, str] = {
    "e": "evict N tokens from head of middle",
    "r": "retrieve LEN tokens from chunk CID at offset OFS",
    "q": "query refs of chunk CID",
    "a": "annotate chunk CID with TAG",
    "l": "link chunk A -> chunk B",
    "s": "top-K similarity search for QUERY (embeddings)",
}


PREFIX_SCHEMA: str = (
    "OPS: e N | r CID OFS LEN | q CID | a CID TAG | l A B | s Q K\n"
    "Window cap L. Issue ops one per line. Use r to bring back evicted tokens.\n"
    "Evict before retrieving when full."
)
