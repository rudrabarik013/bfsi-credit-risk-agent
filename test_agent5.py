from crewai import Crew, Process, Task
from agents.evaluator import risk_evaluator_agent

# ─── Mock context — simulates what Agents 1-4 would have produced ─────────────
MOCK_CONTEXT = """
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

--- AGENT 2 SUMMARY (Data Validator) ---
Data Quality Score: 8/10
Anomalies flagged:
  - Age 22 with little savings and 5951 DM credit is an unusual combination
  - Loan duration of 48 months is long for a radio/TV purchase
No missing fields detected.

--- AGENT 3 SUMMARY (Market Analyst) ---
Macro Risk Level: MODERATE
  - Unemployment Rate: 4.3% (LOW risk)
  - Inflation YoY: 3.29% (LOW-MODERATE risk)
  - Real GDP: 24055 Billion USD (MODERATE risk)
  - Federal Funds Rate: 3.64% (MODERATE risk)
Policy Recommendation: Maintain current lending criteria.

--- AGENT 4 SUMMARY (Compliance Officer) ---
Compliance Verdict: NON-COMPLIANT
Total Violations: 2
  VIOLATION 1: Credit Amount 5951 DM exceeds 5000 DM limit for 'little' savings
  VIOLATION 2: Duration 48 months exceeds 36-month maximum for radio/TV loans
"""

# ─── Risk Evaluation Task ─────────────────────────────────────────────────────
risk_task = Task(
    description=(
        f"You are the Senior Credit Risk Evaluator. Here is the full assessment "
        f"context from all previous agents:\n\n{MOCK_CONTEXT}\n\n"
        "Synthesize ALL of this into a final risk verdict.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 1 — RISK FACTOR SCORECARD\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Score each factor as LOW / MODERATE / HIGH risk:\n\n"
        "  APPLICANT FACTORS:\n"
        "  ┌─────────────────────────┬───────────────┬──────────────────────────┐\n"
        "  │ Factor                  │ Value         │ Risk Level               │\n"
        "  ├─────────────────────────┼───────────────┼──────────────────────────┤\n"
        "  │ Age                     │ 22 years      │ LOW / MODERATE / HIGH    │\n"
        "  │ Employment Stability    │ Skilled       │ LOW / MODERATE / HIGH    │\n"
        "  │ Savings Level           │ little        │ LOW / MODERATE / HIGH    │\n"
        "  │ Checking Account        │ moderate      │ LOW / MODERATE / HIGH    │\n"
        "  │ Credit Amount           │ 5951 DM       │ LOW / MODERATE / HIGH    │\n"
        "  │ Loan Duration           │ 48 months     │ LOW / MODERATE / HIGH    │\n"
        "  │ Loan Purpose            │ radio/TV      │ LOW / MODERATE / HIGH    │\n"
        "  │ Data Quality Score      │ 8/10          │ LOW / MODERATE / HIGH    │\n"
        "  └─────────────────────────┴───────────────┴──────────────────────────┘\n\n"
        "  EXTERNAL FACTORS:\n"
        "  ┌─────────────────────────┬───────────────┬──────────────────────────┐\n"
        "  │ Factor                  │ Value         │ Risk Level               │\n"
        "  ├─────────────────────────┼───────────────┼──────────────────────────┤\n"
        "  │ Macro Risk Level        │ MODERATE      │ LOW / MODERATE / HIGH    │\n"
        "  │ RBI Compliance Status   │ NON-COMPLIANT │ LOW / MODERATE / HIGH    │\n"
        "  │ No. of Violations       │ 2             │ LOW / MODERATE / HIGH    │\n"
        "  └─────────────────────────┴───────────────┴──────────────────────────┘\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 2 — FINAL RISK VERDICT\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Overall Risk Verdict  : GOOD or BAD (choose one)\n"
        "  Confidence Level      : LOW / MEDIUM / HIGH\n"
        "  Primary Risk Drivers  : [top 3 reasons]\n"
        "  Mitigating Factors    : [any factors that reduce risk]\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STEP 3 — RECOMMENDATION\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Decision      : APPROVE / REJECT / CONDITIONAL APPROVE\n"
        "  Conditions    : (if conditional) list specific conditions\n"
        "  One-line note : For the credit committee record\n\n"
        "IMPORTANT: Verdict must be GOOD or BAD — no middle ground."
    ),
    expected_output=(
        "A structured risk evaluation with completed scorecard table, "
        "GOOD/BAD verdict with confidence and top 3 risk drivers, "
        "and APPROVE/REJECT/CONDITIONAL APPROVE decision with one-line note."
    ),
    agent=risk_evaluator_agent
)

# ─── Mini Crew — Agent 5 only ──────────────────────────────────────────────────
crew = Crew(
    agents=[risk_evaluator_agent],
    tasks=[risk_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=25
)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AGENT 5 ISOLATED TEST — Senior Credit Risk Evaluator")
    print("  Applicant ID: 1 | Ground Truth: BAD")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print("  AGENT 5 TEST COMPLETE")
    print("  *** Compare verdict above against Ground Truth: BAD ***")
    print("="*60 + "\n")
