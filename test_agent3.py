from crewai import Crew, Process, Task
from agents.market_analyst import market_analyst_agent

# ─── Market Analysis Task (Agent 3 only) ──────────────────────────────────────
market_analysis_task = Task(
    description=(
        "You are the Market & Economic Risk Analyst. Your job is to assess "
        "the current macroeconomic environment and its implications for "
        "credit risk in the BFSI sector.\n\n"
        "Step 1 — USE YOUR TOOL (call it EXACTLY ONCE):\n"
        "Call the 'FRED Macroeconomic Data Fetcher' tool ONE time only, "
        "using 'fetch all indicators' as the query string. The tool will "
        "return ALL four indicators in a single response. Do NOT call the "
        "tool multiple times or with individual series names.\n\n"
        "Step 2 — INTERPRET EACH INDICATOR:\n"
        "For each indicator, state:\n"
        "  - The current value and what it means\n"
        "  - Whether it represents LOW, MODERATE, or HIGH macro risk\n"
        "  - A one-line implication for credit lending decisions\n\n"
        "Step 3 — OVERALL MACRO RISK VERDICT:\n"
        "Combine all four indicators into a single macro risk assessment:\n"
        "  - Overall Macro Risk Level: LOW / MODERATE / HIGH\n"
        "  - A 2-3 sentence summary of the economic environment\n"
        "  - One recommendation for credit policy (e.g., tighten criteria, "
        "    maintain current thresholds, or relax for certain segments)\n\n"
        "IMPORTANT: Base your assessment ONLY on the data returned by your "
        "tool. Do not guess or hallucinate values. If a value shows 'Error', "
        "note it clearly and assess based on the remaining indicators."
    ),
    expected_output=(
        "A structured macroeconomic risk report with: current values of all "
        "4 FRED indicators, individual risk levels and credit implications "
        "for each, an overall macro risk verdict (LOW/MODERATE/HIGH), "
        "a 2-3 sentence economic summary, and one credit policy recommendation."
    ),
    agent=market_analyst_agent
)

# ─── Mini Crew — Agent 3 only ──────────────────────────────────────────────────
crew = Crew(
    agents=[market_analyst_agent],
    tasks=[market_analysis_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=25
)

# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AGENT 3 ISOLATED TEST — Market Analyst (FRED API)")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print("  AGENT 3 TEST COMPLETE")
    print("="*60 + "\n")
