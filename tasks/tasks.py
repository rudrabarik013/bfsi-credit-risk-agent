from crewai import Task
from agents.collector import data_collector_agent, get_applicant_data
from agents.validator import data_validator_agent
from agents.market_analyst import market_analyst_agent

def create_collection_task(applicant_id: int) -> Task:
    """
    Creates the data collection task for a given applicant ID.
    The Risk label is deliberately excluded from the agent's input
    so the agent reasons purely from the applicant's financial profile.
    """
    applicant_data, _ = get_applicant_data(applicant_id)

    return Task(
        description=(
            f"You have been given the following loan applicant profile "
            f"extracted from our BFSI dataset:\n\n{applicant_data}\n\n"
            f"IMPORTANT: This dataset contains ONLY the following 9 fields:\n"
            f"Age, Sex, Job Type, Housing, Saving Accounts, Checking Account, "
            f"Credit Amount, Duration, and Purpose.\n\n"
            f"Fields like income, credit score, employment history, and marital "
            f"status are NOT part of this dataset. Do NOT flag them as missing — "
            f"they simply do not exist in this data source.\n\n"
            f"Your job is to:\n"
            f"1. Present all 9 available fields in a clean structured format\n"
            f"2. Flag ONLY fields from the above 9 that show NaN or N/A values\n"
            f"3. Note what each available field tells us as a proxy indicator "
            f"   (e.g. Job Type = Skilled suggests stable income)\n"
            f"4. End with a strictly neutral one-line factual summary of what "
            f"   the data shows. Do NOT use words like 'stable', 'positive', "
            f"   'good' or 'concerning'. Just state the facts as observed. "
            f"   Risk judgement is strictly the Risk Evaluator Agent's job."
        ),
        expected_output=(
            "A clean structured summary of all 9 available applicant fields, "
            "with proxy interpretations for each field, any NaN values flagged, "
            "and a neutral one-line first impression. No risk verdict."
        ),
        agent=data_collector_agent
    )


def create_market_analysis_task() -> Task:
    """
    Creates the macroeconomic market analysis task.
    Agent 3 uses the FRED API tool to fetch live indicators
    and interprets their implications for credit risk.
    """
    return Task(
        description=(
            "You are the Market & Economic Risk Analyst. Your job is to assess "
            "the current macroeconomic environment and its implications for "
            "credit risk in the BFSI sector.\n\n"
            "Step 1 — USE YOUR TOOL:\n"
            "Call the 'FRED Macroeconomic Data Fetcher' tool to retrieve the "
            "latest values for: Unemployment Rate, Inflation (CPI), Real GDP, "
            "and Federal Funds Rate.\n\n"
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


def create_validation_task() -> Task:
    """
    Creates the data validation task.
    Two-layer validation architecture:
      Layer 1 — Python hard rules (via tool): range checks, allowed-value checks, missing values
      Layer 2 — LLM reasoning: consistency checks, anomaly detection, imputation suggestions
    Agent 2 receives Agent 1's output automatically via CrewAI context passing.
    """
    return Task(
        description=(
            "You have received an applicant profile from the Data Collector Agent.\n\n"
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
            "  - Does the age match the financial profile? (e.g. 22-year-old "
            "    with 'rich' savings is unusual — flag it)\n"
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
            "A structured two-layer validation report:\n"
            "Section 1 — Hard Validation Results (from tool): PASS/FAIL for "
            "missing values, range checks, and allowed-value checks.\n"
            "Section 2 — Consistency & Anomaly Check (LLM reasoning): "
            "any logical inconsistencies flagged, imputed values marked [IMPUTED].\n"
            "Section 3 — Data Quality Score out of 10 with breakdown and "
            "one-line summary."
        ),
        agent=data_validator_agent
    )
