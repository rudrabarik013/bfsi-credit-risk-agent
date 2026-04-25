from crewai import Task
from agents.collector import data_collector_agent, get_applicant_data
from agents.validator import data_validator_agent
from agents.market_analyst import market_analyst_agent
from agents.compliance import compliance_agent
from agents.evaluator import risk_evaluator_agent
from agents.reporter import report_writer_agent

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


def create_compliance_task(applicant_id: int) -> Task:
    """
    Creates the RBI compliance check task.
    Agent 4 uses RAG to retrieve relevant RBI rules and checks each
    field of the applicant profile for regulatory violations.
    Applicant profile is injected directly to avoid hallucination.
    """
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"You are the RBI Regulatory Compliance Officer. Here is the loan "
            f"applicant profile you must check:\n\n{applicant_data}\n\n"
            "Your job is to check this profile against RBI credit compliance "
            "guidelines using your RAG retrieval tool.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STEP 1 — RETRIEVE RULES (use tool 3 times only)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Call the 'RBI Compliance Rules Retriever' tool exactly 3 times:\n"
            "  1. 'age eligibility minimum maximum loan borrower'\n"
            "  2. 'credit amount limit savings account duration purpose'\n"
            "  3. 'gender sex discrimination employment unskilled'\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STEP 2 — CHECK EACH FIELD AGAINST RETRIEVED RULES\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "For each field, state:\n"
            "  - The applicant's value\n"
            "  - The applicable RBI rule (retrieved above)\n"
            "  - Status: ✅ COMPLIANT or ❌ VIOLATION\n"
            "  - If VIOLATION: quote the exact rule being breached\n\n"
            "Check these fields using retrieved rules EXACTLY:\n\n"
            "  CHECK 1 — Age: Compare applicant's age against RBI minimum (21).\n"
            "    If age >= 21, it is COMPLIANT. Do NOT flag age >= 21 as violation.\n\n"
            "  CHECK 2 — Job Type: Verify employment type meets RBI requirements.\n\n"
            "  CHECK 3 — Credit Amount vs Savings: Compare credit amount against\n"
            "    the savings-ratio limit for the applicant's savings category.\n"
            "    Flag as VIOLATION only if credit amount exceeds the limit.\n\n"
            "  CHECK 4 — Duration vs Purpose: Compare loan duration against the\n"
            "    maximum allowed months for the stated loan purpose.\n"
            "    Flag as VIOLATION only if duration exceeds the limit.\n\n"
            "  CHECK 5 — Sex/Gender: The non-discrimination rule means gender must\n"
            "    NOT be used as a risk factor in the lending DECISION.\n"
            "    Recording gender as data is NOT a violation. Mark as COMPLIANT.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STEP 3 — FINAL COMPLIANCE VERDICT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  - Total violations found: X\n"
            "  - Overall verdict: COMPLIANT or NON-COMPLIANT\n"
            "  - One-line recommendation for the credit officer\n\n"
            "IMPORTANT: Base every judgement ONLY on rules retrieved from "
            "your tool. Currency is DM, not rupees."
        ),
        expected_output=(
            "A structured RBI compliance report with: rules retrieved per "
            "field, COMPLIANT/VIOLATION status for each field with rule "
            "citations, total violation count, overall COMPLIANT/NON-COMPLIANT "
            "verdict, and one-line recommendation for the credit officer."
        ),
        agent=compliance_agent
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


def create_risk_evaluation_task(applicant_id: int) -> Task:
    """
    Creates the final risk evaluation task.
    Agent 5 synthesizes outputs from all 4 previous agents and delivers
    a final GOOD / BAD credit risk verdict with full justification.
    """
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"You are the Senior Credit Risk Evaluator. Here is the applicant "
            f"profile under assessment:\n\n{applicant_data}\n\n"
            "You have received reports from 4 specialist agents before you:\n"
            "  - Agent 1: Data Collector (applicant profile)\n"
            "  - Agent 2: Data Validator (data quality score & anomalies)\n"
            "  - Agent 3: Market Analyst (macroeconomic risk level)\n"
            "  - Agent 4: Compliance Officer (RBI violations)\n\n"
            "Your job is to synthesize ALL of this into a final risk verdict.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STEP 1 — RISK FACTOR SCORECARD\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Score each factor as LOW / MODERATE / HIGH risk:\n\n"
            "  APPLICANT FACTORS (from Agent 1 & 2):\n"
            "  ┌─────────────────────────┬───────────────┬──────────────────────────┐\n"
            "  │ Factor                  │ Value         │ Risk Level               │\n"
            "  ├─────────────────────────┼───────────────┼──────────────────────────┤\n"
            "  │ Age                     │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Employment Stability    │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Savings Level           │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Checking Account        │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Credit Amount           │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Loan Duration           │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Loan Purpose            │ [value]       │ LOW / MODERATE / HIGH    │\n"
            "  │ Data Quality Score      │ [score]/10    │ LOW / MODERATE / HIGH    │\n"
            "  └─────────────────────────┴───────────────┴──────────────────────────┘\n\n"
            "  EXTERNAL FACTORS (from Agent 3 & 4):\n"
            "  ┌─────────────────────────┬───────────────┬──────────────────────────┐\n"
            "  │ Factor                  │ Value         │ Risk Level               │\n"
            "  ├─────────────────────────┼───────────────┼──────────────────────────┤\n"
            "  │ Macro Risk Level        │ [level]       │ LOW / MODERATE / HIGH    │\n"
            "  │ RBI Compliance Status   │ [status]      │ LOW / MODERATE / HIGH    │\n"
            "  │ No. of Violations       │ [count]       │ LOW / MODERATE / HIGH    │\n"
            "  └─────────────────────────┴───────────────┴──────────────────────────┘\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STEP 2 — FINAL RISK VERDICT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Overall Risk Verdict  : GOOD / BAD\n"
            "  Confidence Level      : LOW / MEDIUM / HIGH\n"
            "  Primary Risk Drivers  : [top 3 reasons for your verdict]\n"
            "  Mitigating Factors    : [any factors that reduce risk]\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "STEP 3 — RECOMMENDATION\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Decision       : APPROVE / REJECT / CONDITIONAL APPROVE\n"
            "  Conditions     : (if conditional) list specific conditions\n"
            "  One-line note  : For the credit committee record\n\n"
            "IMPORTANT: Your verdict must be either GOOD or BAD — no middle ground. "
            "Use all previous agent outputs visible in your context to justify it."
        ),
        expected_output=(
            "A structured risk evaluation with: completed Risk Factor Scorecard "
            "table, final GOOD/BAD verdict with confidence level and top 3 risk "
            "drivers, and a clear APPROVE/REJECT/CONDITIONAL APPROVE decision "
            "with one-line note for the credit committee."
        ),
        agent=risk_evaluator_agent
    )


def create_report_task(applicant_id: int) -> Task:
    """
    Creates the final report writing task.
    Agent 6 compiles all previous agent outputs into a single professional
    Credit Assessment Report for the credit committee.
    """
    applicant_data, _ = get_applicant_data(applicant_id)
    return Task(
        description=(
            f"You are the Credit Assessment Report Writer. Compile a final, "
            f"professional Credit Assessment Report for the following applicant:\n\n"
            f"{applicant_data}\n\n"
            "Use ALL findings from the previous 5 agents visible in your context. "
            "Write the report in the exact structure below:\n\n"

            "════════════════════════════════════════════════════════════\n"
            "        BFSI CREDIT RISK ASSESSMENT REPORT\n"
            "════════════════════════════════════════════════════════════\n"
            "Report Date        : [today's date]\n"
            "Applicant ID       : [ID]\n"
            "Assessment System  : BFSI Multi-Agent Credit Risk System v1.0\n"
            "────────────────────────────────────────────────────────────\n\n"

            "SECTION 1 — APPLICANT PROFILE SUMMARY\n"
            "Present all 9 fields in a clean two-column format.\n\n"

            "SECTION 2 — DATA QUALITY ASSESSMENT\n"
            "  Data Quality Score    : X/10\n"
            "  Missing Fields        : None / [list]\n"
            "  Anomalies Flagged     : [list or None]\n"
            "  Assessment            : [one line]\n\n"

            "SECTION 3 — MACROECONOMIC CONTEXT\n"
            "  Macro Risk Level      : LOW / MODERATE / HIGH\n"
            "  Key Indicators        : [unemployment, inflation, GDP, rate]\n"
            "  Credit Policy Note    : [one line from Agent 3]\n\n"

            "SECTION 4 — RBI COMPLIANCE STATUS\n"
            "  Overall Status        : COMPLIANT / NON-COMPLIANT\n"
            "  Violations Found      : [count]\n"
            "  Violation Details     : [list each violation clearly]\n\n"

            "SECTION 5 — RISK EVALUATION SUMMARY\n"
            "  ┌─────────────────────────┬───────────────┬────────────┐\n"
            "  │ Factor                  │ Value         │ Risk Level │\n"
            "  ├─────────────────────────┼───────────────┼────────────┤\n"
            "  │ Age                     │ [value]       │ [level]    │\n"
            "  │ Employment Stability    │ [value]       │ [level]    │\n"
            "  │ Savings Level           │ [value]       │ [level]    │\n"
            "  │ Checking Account        │ [value]       │ [level]    │\n"
            "  │ Credit Amount           │ [value]       │ [level]    │\n"
            "  │ Loan Duration           │ [value]       │ [level]    │\n"
            "  │ Loan Purpose            │ [value]       │ [level]    │\n"
            "  │ Macro Environment       │ [value]       │ [level]    │\n"
            "  │ Compliance Status       │ [value]       │ [level]    │\n"
            "  └─────────────────────────┴───────────────┴────────────┘\n\n"
            "  Overall Risk Verdict  : GOOD / BAD\n"
            "  Confidence Level      : LOW / MEDIUM / HIGH\n"
            "  Primary Risk Drivers  : [top 3 reasons]\n\n"

            "SECTION 6 — FINAL DECISION\n"
            "  ┌─────────────────────────────────────────────────────┐\n"
            "  │  DECISION: APPROVE / REJECT / CONDITIONAL APPROVE   │\n"
            "  └─────────────────────────────────────────────────────┘\n"
            "  Rationale  : [2-3 sentences explaining the decision]\n"
            "  Conditions : [if conditional — list conditions, else N/A]\n\n"

            "────────────────────────────────────────────────────────────\n"
            "DISCLAIMER: This report is generated by an AI-based multi-agent\n"
            "system for decision-support purposes only. Final credit decisions\n"
            "must be reviewed and approved by a qualified credit officer as per\n"
            "RBI guidelines.\n"
            "════════════════════════════════════════════════════════════\n\n"

            "IMPORTANT: Fill in every [placeholder] with actual values from "
            "previous agent outputs. Do not leave any section blank."
        ),
        expected_output=(
            "A complete, professional Credit Assessment Report with all 6 sections "
            "filled in — applicant profile, data quality, macro context, compliance "
            "status, risk scorecard, and final decision with rationale."
        ),
        agent=report_writer_agent
    )
