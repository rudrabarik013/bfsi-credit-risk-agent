from crewai import Agent
from crewai.tools import tool
from agents.llm_config import llm


# ─── Hard-coded Python Validation Rules ───────────────────────────────────────
# These rules are 100% deterministic — no LLM involved.
# The LLM handles only consistency & anomaly detection (Step 2).

VALID_JOB_TYPES = {
    "unskilled (non-resident)",
    "unskilled (resident)",
    "skilled",
    "highly skilled",
}

VALID_HOUSING = {"own", "rent", "free"}

VALID_SAVING_ACCOUNTS = {"little", "moderate", "quite rich", "rich", "na", "nan", "n/a"}

VALID_CHECKING_ACCOUNT = {"little", "moderate", "rich", "na", "nan", "n/a"}


@tool("Hard Validation Rules Checker")
def hard_validation_tool(applicant_profile: str) -> str:
    """
    Runs deterministic Python-based hard validation rules on an applicant profile string.

    Checks performed:
    1. Missing value detection (NaN, N/A, empty)
    2. Range checks: Age (18–100), Credit Amount (>0), Duration (>0)
    3. Allowed-value checks: Job Type, Housing, Saving Accounts, Checking Account

    Input: the raw applicant profile text (paste the full profile).
    Output: a structured PASS/FAIL report for each field — 100% rule-based, no LLM reasoning.
    """
    lines = applicant_profile.strip().splitlines()
    parsed = {}

    # Parse key-value pairs from the profile string
    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[str(key).strip()] = str(value).strip()

    results = []
    results.append("HARD VALIDATION REPORT (Python Rule Engine — 100% Deterministic)")
    results.append("=" * 65)

    # ── 1. MISSING VALUE CHECK ─────────────────────────────────────────────────
    results.append("\n[STEP 1] MISSING VALUE CHECK")
    results.append("-" * 40)
    missing_flags = []
    nan_tokens = {"nan", "n/a", "na", "none", "", "null"}

    for field, value in parsed.items():
        if value.lower() in nan_tokens:
            missing_flags.append(field)
            results.append(f"  ❌ MISSING  — {field}: '{value}'  → [NEEDS IMPUTATION]")

    if not missing_flags:
        results.append("  ✅ All fields present — no missing values detected.")

    # ── 2. RANGE CHECK ────────────────────────────────────────────────────────
    results.append("\n[STEP 2] RANGE & FORMAT CHECK")
    results.append("-" * 40)

    # Age
    age_raw = parsed.get("Age", "N/A")
    try:
        age = float(age_raw)
        if 18 <= age <= 100:
            results.append(f"  ✅ VALID    — Age: {age_raw} (within 18–100)")
        else:
            results.append(f"  ❌ INVALID  — Age: {age_raw} (must be 18–100)")
    except ValueError:
        results.append(f"  ❌ INVALID  — Age: '{age_raw}' (not a number)")

    # Credit Amount
    credit_raw = parsed.get("Credit Amount", "N/A").replace("DM", "").strip()
    try:
        credit = float(credit_raw)
        if credit > 0:
            results.append(f"  ✅ VALID    — Credit Amount: {credit_raw} DM (positive number)")
        else:
            results.append(f"  ❌ INVALID  — Credit Amount: {credit_raw} DM (must be > 0)")
    except ValueError:
        results.append(f"  ❌ INVALID  — Credit Amount: '{credit_raw}' (not a number)")

    # Duration
    duration_raw = parsed.get("Duration", "N/A").replace("months", "").strip()
    try:
        duration = float(duration_raw)
        if duration > 0:
            results.append(f"  ✅ VALID    — Duration: {duration_raw} months (positive number)")
        else:
            results.append(f"  ❌ INVALID  — Duration: {duration_raw} months (must be > 0)")
    except ValueError:
        results.append(f"  ❌ INVALID  — Duration: '{duration_raw}' (not a number)")

    # ── 3. ALLOWED-VALUE CHECK ────────────────────────────────────────────────
    results.append("\n[STEP 3] ALLOWED-VALUE CHECK")
    results.append("-" * 40)

    job_raw = parsed.get("Job Type", "N/A")
    if job_raw.lower() in VALID_JOB_TYPES:
        results.append(f"  ✅ VALID    — Job Type: '{job_raw}'")
    else:
        results.append(f"  ❌ INVALID  — Job Type: '{job_raw}' (not in allowed list)")

    housing_raw = parsed.get("Housing", "N/A")
    if housing_raw.lower() in VALID_HOUSING:
        results.append(f"  ✅ VALID    — Housing: '{housing_raw}'")
    else:
        results.append(f"  ❌ INVALID  — Housing: '{housing_raw}' (not in allowed list)")

    savings_raw = parsed.get("Saving Accounts", "N/A")
    if savings_raw.lower() in VALID_SAVING_ACCOUNTS:
        results.append(f"  ✅ VALID    — Saving Accounts: '{savings_raw}'")
    else:
        results.append(f"  ❌ INVALID  — Saving Accounts: '{savings_raw}' (not in allowed list)")

    checking_raw = parsed.get("Checking Account", "N/A")
    if checking_raw.lower() in VALID_CHECKING_ACCOUNT:
        results.append(f"  ✅ VALID    — Checking Account: '{checking_raw}'")
    else:
        results.append(f"  ❌ INVALID  — Checking Account: '{checking_raw}' (not in allowed list)")

    results.append("\n" + "=" * 65)
    results.append("Hard validation complete. Proceed to LLM consistency check.")

    return "\n".join(results)


# ─── Define the Data Validator Agent ─────────────────────────────────────────
data_validator_agent = Agent(
    role="BFSI Data Quality Specialist",

    goal=(
        "Validate and clean the applicant profile received from the Data "
        "Collector. Ensure data integrity before it flows to analysis agents."
    ),

    backstory=(
        "You are a data governance expert with 10 years of experience in BFSI "
        "data quality management. You have worked with RBI-regulated banks where "
        "data accuracy is non-negotiable for credit decisions. You know that a "
        "single missing or incorrect field can lead to a wrong credit decision "
        "worth lakhs of rupees. You are systematic, thorough, and never let "
        "bad data pass through to downstream teams."
    ),

    tools=[hard_validation_tool],
    llm=llm,
    verbose=True
)
