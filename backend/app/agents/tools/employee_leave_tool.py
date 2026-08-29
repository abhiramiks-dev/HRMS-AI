"""Structured demo tool for fictional employee leave information."""

import re
from typing import TypedDict


class EmployeeLeaveRecord(TypedDict):
    """Leave fields available in the fictional demo data."""

    employee_id: str
    employee_name: str
    years_of_experience: int
    annual_leave_days: int
    emergency_leave_days: int


# DEMO DATA ONLY: replace with an HR data service when one is available.
DEMO_EMPLOYEE_LEAVE_DATA: dict[str, EmployeeLeaveRecord] = {
    "E001": {
        "employee_id": "E001",
        "employee_name": "Aisha Khan",
        "years_of_experience": 5,
        "annual_leave_days": 21,
        "emergency_leave_days": 5,
    },
    "E002": {
        "employee_id": "E002",
        "employee_name": "Daniel Lee",
        "years_of_experience": 3,
        "annual_leave_days": 18,
        "emergency_leave_days": 4,
    },
    "E003": {
        "employee_id": "E003",
        "employee_name": "Maya Patel",
        "years_of_experience": 8,
        "annual_leave_days": 25,
        "emergency_leave_days": 6,
    },
}


class EmployeeLeaveTool:
    """Look up fictional employee leave records by employee ID."""

    EMPLOYEE_ID_PATTERN = re.compile(r"\bE\d{3}\b", re.IGNORECASE)

    def __init__(
        self,
        records: dict[str, EmployeeLeaveRecord] | None = None,
    ) -> None:
        """Configure the tool with injected or default demo records."""
        self._records = records if records is not None else DEMO_EMPLOYEE_LEAVE_DATA

    def answer(self, question: str) -> str:
        """Return a readable leave summary or an employee-not-found message."""
        match = self.EMPLOYEE_ID_PATTERN.search(question)
        if match is None:
            return "No employee ID was provided."

        employee_id = match.group(0).upper()
        record = self._records.get(employee_id)
        if record is None:
            return f"Employee {employee_id} was not found in the demo data."

        return (
            f"{record['employee_name']} ({employee_id}) has "
            f"{record['annual_leave_days']} annual leave days and "
            f"{record['emergency_leave_days']} emergency leave days."
        )
