from crewai import Task
from agents.collector import data_collector_agent, get_applicant_data

def create_collection_task(applicant_id: int) -> Task:
    """
    Creates the data collection task for a given applicant ID.
    The Risk label is deliberately excluded from the agent's input
    so the agent reasons purely from the applicant's financial profile.
    """
    # Get the raw applicant data — Risk label is returned separately
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
