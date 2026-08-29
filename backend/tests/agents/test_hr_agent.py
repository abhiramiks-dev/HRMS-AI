"""Tests for deterministic HR agent tool routing."""

from backend.app.agents.hr_agent import HRAgent
from backend.app.agents.tools.employee_leave_tool import EmployeeLeaveTool
from backend.app.agents.tools.policy_search_tool import PolicySearchTool


class FakePolicyTool:
    """Capture policy-tool calls without invoking Gemini."""

    def __init__(self) -> None:
        self.question: str | None = None

    def answer(self, question: str) -> str:
        self.question = question
        return "Policy answer"


class FakeEmployeeTool:
    """Capture employee-tool calls without external services."""

    def __init__(self) -> None:
        self.question: str | None = None

    def answer(self, question: str) -> str:
        self.question = question
        return "Employee answer"


class FakeRAGService:
    """Return a deterministic RAG answer and capture policy questions."""

    def __init__(self) -> None:
        self.question: str | None = None

    def answer_question(self, question: str) -> "FakeRAGAnswer":
        self.question = question
        return FakeRAGAnswer(answer="Annual leave policy answer")


class FakeRAGAnswer:
    """Minimal RAG result shape required by the policy tool."""

    def __init__(self, answer: str) -> None:
        self.answer = answer


def test_policy_question_routes_to_policy_search_tool() -> None:
    policy = FakePolicyTool()
    employee = FakeEmployeeTool()
    question = "What is the annual leave entitlement policy?"

    answer = HRAgent(policy, employee).answer_question(question)

    assert answer == "Policy answer"
    assert policy.question == question
    assert employee.question is None


def test_employee_question_routes_to_employee_leave_tool() -> None:
    policy = FakePolicyTool()
    employee = FakeEmployeeTool()
    question = "How much leave does E001 have?"

    answer = HRAgent(policy, employee).answer_question(question)

    assert answer == "Employee answer"
    assert employee.question == question
    assert policy.question is None


def test_unknown_employee_returns_clear_demo_data_message() -> None:
    answer = EmployeeLeaveTool().answer("Show leave information for E999")

    assert answer == "Employee E999 was not found in the demo data."


def test_policy_search_tool_reuses_injected_rag_service() -> None:
    rag_service = FakeRAGService()
    tool = PolicySearchTool(rag_service)
    question = "How many annual leave days are employees entitled to?"

    assert tool.answer(question) == "Annual leave policy answer"
    assert rag_service.question == question


def test_agent_returns_selected_tool_answer() -> None:
    answer = HRAgent(
        PolicySearchTool(FakeRAGService()),
        EmployeeLeaveTool(),
    ).answer_question("What is the leave policy?")

    assert answer == "Annual leave policy answer"
