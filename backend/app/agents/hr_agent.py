"""Deterministic first-stage tool orchestration for HR questions."""

import re
from typing import Protocol


class AgentTool(Protocol):
    """Interface shared by tools selected by the agent."""

    def answer(self, question: str) -> str:
        """Answer a question handled by the tool."""


class HRAgent:
    """Route HR questions to policy search or employee leave lookup.

    Routing is intentionally deterministic keyword/identifier matching rather
    than simulated LLM reasoning. The injected tool boundary can later be
    replaced by an LLM-based tool-calling router without changing callers.
    """

    EMPLOYEE_ID_PATTERN = re.compile(r"\bE\d{3}\b", re.IGNORECASE)

    def __init__(
        self,
        policy_search_tool: AgentTool,
        employee_leave_tool: AgentTool,
    ) -> None:
        """Configure the agent with its available tools."""
        self._policy_search_tool = policy_search_tool
        self._employee_leave_tool = employee_leave_tool

    def answer_question(self, question: str) -> str:
        """Select the appropriate tool and return its answer."""
        if self.EMPLOYEE_ID_PATTERN.search(question):
            return self._employee_leave_tool.answer(question)
        return self._policy_search_tool.answer(question)
