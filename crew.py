from crewai import Crew, Process
from agents.collector import data_collector_agent, get_applicant_data
from agents.validator import data_validator_agent
from agents.market_analyst import market_analyst_agent
from agents.compliance import compliance_agent
from agents.evaluator import risk_evaluator_agent
from agents.reporter import report_writer_agent
from tasks.tasks import (
    create_collection_task,
    create_validation_task,
    create_market_analysis_task,
    create_compliance_task,
    create_risk_evaluation_task,
    create_report_task,
)

# ─── Choose which applicant to assess ────────────────────────────────────────
APPLICANT_ID = 1

# ─── Get actual risk label for benchmarking ONLY ─────────────────────────────
_, actual_risk_label = get_applicant_data(APPLICANT_ID)

# ─── Create tasks ─────────────────────────────────────────────────────────────
collection_task      = create_collection_task(APPLICANT_ID)
validation_task      = create_validation_task()
market_analysis_task = create_market_analysis_task()
compliance_task      = create_compliance_task(APPLICANT_ID)
risk_evaluation_task = create_risk_evaluation_task(APPLICANT_ID)
report_task          = create_report_task(APPLICANT_ID)

# ─── Assemble the Full Crew (All 6 Agents) ────────────────────────────────────
crew = Crew(
    agents=[
        data_collector_agent,
        data_validator_agent,
        market_analyst_agent,
        compliance_agent,
        risk_evaluator_agent,
        report_writer_agent,
    ],
    tasks=[
        collection_task,
        validation_task,
        market_analysis_task,
        compliance_task,
        risk_evaluation_task,
        report_task,
    ],
    process=Process.sequential,
    verbose=True,
    max_rpm=25
)

# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BFSI CREDIT RISK AGENT — FULL PIPELINE")
    print(f"  Assessing Applicant ID : {APPLICANT_ID}")
    print(f"  Total Agents           : 6")
    print("="*60 + "\n")

    result = crew.kickoff()

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  BENCHMARK : Actual Risk Label = '{actual_risk_label.upper()}'")
    print(f"  Compare this against Agent 5's verdict above.")
    print("="*60 + "\n")
