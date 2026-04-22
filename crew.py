from crewai import Crew, Process
from agents.collector import data_collector_agent, get_applicant_data
from tasks.tasks import create_collection_task

# ─── Choose which applicant to assess ────────────────────────────────────────
# Change this number (0 to 999) to test different applicants
APPLICANT_ID = 1

# ─── Get the actual risk label for benchmarking ONLY ─────────────────────────
# This is kept hidden from all agents — used only at the end to check accuracy
_, actual_risk_label = get_applicant_data(APPLICANT_ID)

# ─── Create the task for this applicant ──────────────────────────────────────
collection_task = create_collection_task(APPLICANT_ID)

# ─── Assemble the Crew (only Agent 1 for now) ────────────────────────────────
crew = Crew(
    agents=[data_collector_agent],
    tasks=[collection_task],
    process=Process.sequential,
    verbose=True
)

# ─── Run it! ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BFSI CREDIT RISK AGENT — STARTING")
    print(f"  Assessing Applicant ID: {APPLICANT_ID}")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print("  AGENT 1 OUTPUT:")
    print("="*60)
    print(result)

    # Show the ground truth ONLY after the agent has completed its work
    print("\n" + "="*60)
    print(f"  BENCHMARK: Actual Risk Label from Dataset = '{actual_risk_label.upper()}'")
    print("  (This will be compared against Agent 5's independent assessment)")
    print("="*60 + "\n")
