from typing import Any

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Conversation and workflow state for one DevPulse user."""

    user_id: str
    user_preferences: dict[str, Any]
    awaiting_approval: bool
    completed_tools: list[str]
    pending_email: dict[str, str] | None
    pending_slack: dict[str, str] | None
    approval_result: str | None
