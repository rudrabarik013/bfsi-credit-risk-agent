from crewai import Crew, Process, Task
from agents.validator import data_validator_agent

# ─── Hardcoded Agent 1 output (simulates what Agent 1 passes to Agent 2) ──────
# This is the same format Agent 1 produces for Applicant ID 1
MOCK_AGENT1_OUTPUT = """
APPLICANT PROFILE (ID: 1)
=======================================
Age              : 22
Sex              : male
Job Type         : Skilled
Housing          : own
Saving Accounts  : little
Checking Account : moderate
Credit Amount    : 5951 DM
Duration         : 48 months
Purpose          : radio/TV
"""

# ─── Validation Task (Agent 2 only) ───────────────────────────────────────────
validation_task = Task(
    description=(
        f"You have received the following applicant profile from the Data Collector Agent:\n\n"
        f"{MOCK_AGENT1_OUTPUT}\n\n"
        "You must follow a TWO-STEP validation process:\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 1 — HARD VALIDATION (USE YOUR TOOL — MANDATORY)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Call the 'Hard Validation Rules Checker' tool, passing the full "
        "applicant profile as input.\n"
        "This tool runs deterministic Python rules for:\n"
        "  - Missing value detection (NaN, N/A, null)\n"
        "  - Range checks: Age (18–100), Credit Amount (>0), Duration (>0)\n"
        "  - Allowed-value checks: Job Type, Housing, Saving Accounts, Checking Account\n"
        "Report the tool's PASS/FAIL results exactly as returned — do NOT "
        "override or second-guess them. These are rule-based facts.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 2 — CONSISTENCY & ANOMALY CHECK (YOUR LLM REASONING)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Now apply your expert judgment to check if the data makes "
        "logical sense as a whole. Consider:\n"
        "  - Does the age match the financial profile?\n"
        "  - Does the loan amount match the savings/checking level?\n"
        "  - Does the loan duration make sense for the purpose?\n"
        "  - Are there any unusual combinations across the 9 fields?\n"
        "For any missing fields found in Step 1, suggest a reasonable "
        "imputed value based on context and mark it clearly as [IMPUTED].\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 3 — DATA QUALITY SCORE\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Give a final Data Quality Score out of 10. Scoring guide:\n"
        "  - Start at 10\n"
        "  - Deduct 1 point per missing field (Step 1)\n"
        "  - Deduct 1 point per failed range/allowed-value check (Step 1)\n"
        "  - Deduct 0.5 points per consistency anomaly flagged (Step 2)\n"
        "Show the score breakdown clearly, then give a one-line summary."
    ),
    expected_output=(
        "A structured two-layer validation report with hard validation results, "
        "consistency check results, and a Data Quality Score out of 10."
    ),
    agent=data_validator_agent
)

# ─── Mini Crew — Agent 2 only ──────────────────────────────────────────────────
crew = Crew(
    agents=[data_validator_agent],
    tasks=[validation_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=25
)

# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AGENT 2 ISOLATED TEST — Data Validator")
    print("  Using hardcoded Applicant ID 1 profile")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print("  AGENT 2 TEST COMPLETE")
    print("="*60 + "\n")
