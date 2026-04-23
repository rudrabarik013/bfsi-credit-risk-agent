from crewai import Crew, Process
from agents.collector import data_collector_agent, get_applicant_data
from agents.validator import data_validator_agent
from agents.market_analyst import market_analyst_agent
from tasks.tasks import (
    create_collection_task,
    create_validation_task,
    create_market_analysis_task,
)

# ─── Choose which applicant to assess ────────────────────────────────────────
APPLICANT_ID = 1

# ─── Get actual risk label for benchmarking ONLY ─────────────────────────────
_, actual_risk_label = get_applicant_data(APPLICANT_ID)

# ─── Create tasks ─────────────────────────────────────────────────────────────
collection_task      = create_collection_task(APPLICANT_ID)
validation_task      = create_validation_task()
market_analysis_task = create_market_analysis_task()

# ─── Assemble the Crew (Agents 1, 2 & 3) ─────────────────────────────────────
crew = Crew(
    agents=[data_collector_agent, data_validator_agent, market_analyst_agent],
    tasks=[collection_task, validation_task, market_analysis_task],
    process=Process.sequential,
    verbose=True,
    max_rpm=25        # Throttle to 25 requests/min — respects Groq free tier limit
)

# ─── Run it! ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BFSI CREDIT RISK AGENT — STARTING")
    print(f"  Assessing Applicant ID: {APPLICANT_ID}")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print(f"  BENCHMARK: Actual Risk Label = '{actual_risk_label.upper()}'")
    print("  (Reserved for Agent 5 comparison)")
    print("="*60 + "\n")
