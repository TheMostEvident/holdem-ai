"""Common interface for poker agents.

Agents must act only on the acting player's observable RLCard state.
The engine does not expose the opponent's private hole cards.
"""

from typing import Any, Mapping, Protocol


class Agent(Protocol):
    """Any agent implementing select_action can play via HandEngine."""

    def select_action(self, state: Mapping[str, Any]) -> int:
        """Choose one of the integer IDs in state['legal_actions']."""
        ...
