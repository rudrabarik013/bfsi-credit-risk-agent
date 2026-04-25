from crewai import Task
from agents.collector import data_collector_agent, get_applicant_data
from agents.validator import data_validator_agent
from agents.market_analyst import market_analyst_agent
from agents.compliance import compliance_agent
from agents.evaluator import risk_evaluator_agent
from agents.reporter import report_writer_agent


def create_collection_task(applicant_id: int) -> Task:
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"Loan applicant profile from BFSI dataset:\n\n{applicant_data}\n\n"
            f"Dataset has ONLY 9 fields: Age, Sex, Job Type, Housing, Saving Accounts, "
            f"Checking Account, Credit Amount, Duration, Purpose. "
            f"Do NOT flag income, credit score, or employment history as missing — they don't exist here.\n\n"
            f"Your tasks: 1) Present all 9 fields clearly. "
            f"2) Flag any NaN/N/A values among these 9 fields only. "
            f"3) Note what each field tells us as a proxy indicator. "
            f"4) End with a neutral one-line factual summary. No risk verdict."
        ),
        expected_output=(
            "Structured summary of all 9 applicant fields with proxy interpretations, "
            "NaN flags if any, and a neutral one-line summary. No risk verdict."
        ),
        agent=data_collector_agent
    )


def create_validation_task(applicant_id: int) -> Task:
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"Applicant profile to validate:\n\n{applicant_data}\n\n"
            "Step 1 - HARD VALIDATION (mandatory): Call the 'Hard Validation Rules Checker' "
            "tool with the applicant profile above (copy it exactly as-is). "
            "Report its PASS/FAIL results exactly — do not override them.\n\n"
            "Step 2 - CONSISTENCY CHECK: Use your judgment to check if the data makes "
            "logical sense. Flag unusual combinations (e.g. young age + large loan + little savings). "
            "Suggest imputed values for any missing fields, marked [IMPUTED].\n\n"
            "Step 3 - DATA QUALITY SCORE: Score out of 10. "
            "Start at 10, deduct 1 per missing field, 1 per failed hard check, "
            "0.5 per consistency anomaly. Show breakdown and one-line summary."
        ),
        expected_output=(
            "Two-layer validation report: hard validation PASS/FAIL results, "
            "consistency anomalies, imputed values marked [IMPUTED], "
            "and Data Quality Score out of 10 with breakdown."
        ),
        agent=data_validator_agent
    )


def create_market_analysis_task() -> Task:
    return Task(
        description=(
            "You are the Market & Economic Risk Analyst for BFSI credit risk.\n\n"
            "Step 1: Call the 'FRED Macroeconomic Data Fetcher' tool ONCE using "
            "'fetch all indicators' as the query. Do not call it multiple times.\n\n"
            "Step 2: For each of the 4 indicators returned, state the value, "
            "risk level (LOW/MODERATE/HIGH), and one-line credit implication.\n\n"
            "Step 3: Give an overall Macro Risk Level (LOW/MODERATE/HIGH), "
            "a 2-sentence economic summary, and one credit policy recommendation."
        ),
        expected_output=(
            "Macro risk report with 4 FRED indicator values, individual risk levels, "
            "overall verdict (LOW/MODERATE/HIGH), economic summary, and policy recommendation."
        ),
        agent=market_analyst_agent
    )


def create_compliance_task(applicant_id: int) -> Task:
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"You are the RBI Compliance Officer. Applicant profile:\n\n{applicant_data}\n\n"
            "Step 1 - PYTHON CHECK (mandatory first): Call 'Python Hard Compliance Checker' "
            "with the applicant profile. Report results exactly as returned — do not override.\n\n"
            "Step 2 - RAG CONTEXT: Call 'RBI Compliance Rules Retriever' once with query "
            "'credit amount savings duration purpose employment'. Use retrieved text to add "
            "rule citations to any violations found in Step 1.\n\n"
            "Step 3 - VERDICT: State total violations, overall COMPLIANT/NON-COMPLIANT, "
            "and one-line recommendation. Currency is DM, not rupees."
        ),
        expected_output=(
            "Compliance report with Python check results, RBI rule citations for violations, "
            "total violation count, COMPLIANT/NON-COMPLIANT verdict, and recommendation."
        ),
        agent=compliance_agent
    )


def create_risk_evaluation_task(applicant_id: int) -> Task:
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"You are the Senior Credit Risk Evaluator. Applicant:\n\n{applicant_data}\n\n"
            "You have outputs from 4 agents: Data Collector, Validator, Market Analyst, "
            "and Compliance Officer. Synthesize all findings into a final verdict.\n\n"
            "Produce a Risk Factor Scorecard with these factors rated LOW/MODERATE/HIGH:\n"
            "Applicant: Age, Employment, Savings, Checking Account, Credit Amount, "
            "Duration, Purpose, Data Quality Score.\n"
            "External: Macro Risk Level, Compliance Status, No. of Violations.\n\n"
            "Then give: Overall Risk Verdict (GOOD or BAD — no middle ground), "
            "Confidence Level (LOW/MEDIUM/HIGH), top 3 Primary Risk Drivers, "
            "Mitigating Factors, and Decision (APPROVE/REJECT/CONDITIONAL APPROVE) "
            "with a one-line note for the credit committee."
        ),
        expected_output=(
            "Risk Factor Scorecard with all factors rated, GOOD/BAD verdict with "
            "confidence level and top 3 risk drivers, and APPROVE/REJECT/CONDITIONAL "
            "APPROVE decision with one-line committee note."
        ),
        agent=risk_evaluator_agent
    )


def create_report_task(applicant_id: int) -> Task:
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"You are the Credit Assessment Report Writer. Applicant:\n\n{applicant_data}\n\n"
            "Using all previous agent findings, write a professional Credit Assessment Report "
            "with these 6 sections:\n\n"
            "Section 1 - Applicant Profile Summary: all 9 fields in two-column format.\n"
            "Section 2 - Data Quality: score, missing fields, anomalies, one-line assessment.\n"
            "Section 3 - Macro Context: risk level, 4 key indicators, policy note.\n"
            "Section 4 - RBI Compliance: status, violation count, violation details.\n"
            "Section 5 - Risk Scorecard: table of all factors with risk levels, "
            "overall GOOD/BAD verdict, confidence, top 3 risk drivers.\n"
            "Section 6 - Final Decision: APPROVE/REJECT/CONDITIONAL APPROVE, "
            "rationale in 2-3 sentences, conditions if any.\n\n"
            "End with: DISCLAIMER: This report is AI-generated for decision-support only. "
            "Final decisions require qualified credit officer review per RBI guidelines."
        ),
        expected_output=(
            "Complete Credit Assessment Report with all 6 sections filled using actual "
            "values from previous agents — profile, quality, macro, compliance, "
            "risk scorecard, and final decision with rationale."
        ),
        agent=report_writer_agent
    )
