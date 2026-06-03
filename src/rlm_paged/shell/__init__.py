"""Shell-harness: a simpler stateless-turn architecture where each turn
is one model call + a sequence of fenced shell commands executed against
a per-agent walled-off filesystem.

See DESIGN.md and README for the canonical (typed-blocks) architecture.
This module is the deliberately-minimal alternative where the "API" is
just the unix shell and durable state is just files.
"""

from rlm_paged.shell.agent_fs import AGENT_FILES, AgentFS
from rlm_paged.shell.extractor import (
    FencedBlock,
    extract_blocks,
    extract_shell_commands,
)
from rlm_paged.shell.shell_runner import (
    CommandResult,
    ShellRunner,
    ShellSecurityError,
)
from rlm_paged.shell.shell_runner_cell import (
    ShellCell,
    ShellCellResult,
    run_shell_cell,
)
from rlm_paged.shell.system_prompt import (
    SYSTEM_PROMPT_TEMPLATE,
    render_system_prompt,
)

__all__ = [
    "AGENT_FILES",
    "AgentFS",
    "CommandResult",
    "FencedBlock",
    "SYSTEM_PROMPT_TEMPLATE",
    "ShellCell",
    "ShellCellResult",
    "ShellRunner",
    "ShellSecurityError",
    "extract_blocks",
    "extract_shell_commands",
    "render_system_prompt",
    "run_shell_cell",
]
