from crewai import Crew, Process, Task
from agents.compliance import compliance_agent

# ─── Hardcoded Applicant Profile (same as Agent 1 output for ID: 1) ───────────
MOCK_APPLICANT_PROFILE = """
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

# ─── Compliance Task with real profile injected ────────────────────────────────
compliance_task = Task(
    description=(
        f"You are the RBI Regulatory Compliance Officer. Here is the loan "
        f"applicant profile you must check:\n\n{MOCK_APPLICANT_PROFILE}\n\n"
        "Check this profile against RBI compliance guidelines.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 1 — PYTHON HARD CHECK (MANDATORY — call first)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Call the 'Python Hard Compliance Checker' tool with the full "
        "applicant profile. Report its PASS/FAIL results EXACTLY as returned. "
        "Do NOT override or second-guess any result from this tool.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 2 — RAG CONTEXT (call RBI Compliance Rules Retriever once)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Call the 'RBI Compliance Rules Retriever' tool once with query: "
        "'credit amount savings duration purpose employment type'. "
        "Use the retrieved text to add regulatory citations to the violations "
        "found in Step 1.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 3 — FINAL COMPLIANCE VERDICT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Based ONLY on Step 1 results:\n"
        "  - Total violations found\n"
        "  - Overall verdict: COMPLIANT or NON-COMPLIANT\n"
        "  - One-line recommendation for the credit officer"
    ),
    expected_output=(
        "A structured RBI compliance report using the actual applicant values, "
        "with COMPLIANT/VIOLATION status for each field, rule citations, "
        "total violation count, overall verdict, and recommendation."
    ),
    agent=compliance_agent
)

# ─── Mini Crew — Agent 4 only ──────────────────────────────────────────────────
crew = Crew(
    agents=[compliance_agent],
    tasks=[compliance_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=25
)

# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AGENT 4 ISOLATED TEST — RBI Compliance Officer (RAG)")
    print("  Applicant ID: 1 | Age: 22 | Savings: little | 5951 DM | 48 months | radio/TV")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print("  AGENT 4 TEST COMPLETE")
    print("="*60 + "\n")
