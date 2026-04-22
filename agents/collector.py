import os
import pandas as pd
from crewai import Agent, LLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Setup the LLM (Groq - Free) ─────────────────────────────────────────────
# This is the "brain" behind our agent — using LLaMA 3 via Groq (free)
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

# ─── Load the Dataset ─────────────────────────────────────────────────────────
# Get the path to our CSV file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "german_credit_data.csv")

def get_applicant_data(applicant_id: int) -> str:
    """
    Reads the CSV and extracts one applicant row by index.
    Returns a clean, readable string summary of the applicant's profile.
    """
    df = pd.read_csv(DATA_PATH)

    # Make sure the ID is valid
    if applicant_id < 0 or applicant_id >= len(df):
        return f"Error: Applicant ID {applicant_id} not found. Valid range: 0 to {len(df)-1}"

    # Extract the row as a dictionary
    row = df.iloc[applicant_id].to_dict()

    # Store the actual risk label separately for benchmarking only
    # This is NEVER passed to any agent — it's only used at the end
    # to compare Agent 5's independent assessment against the known answer
    actual_risk = row.get('Risk', 'N/A')

    # ── Map numerical Job column to human-readable label ──────────────────────
    job_map = {
        0: "Unskilled (non-resident)",
        1: "Unskilled (resident)",
        2: "Skilled",
        3: "Highly Skilled"
    }
    job_raw = row.get('Job', None)
    try:
        job_label = job_map.get(int(job_raw), f"Unknown (value: {job_raw})")
    except (TypeError, ValueError):
        job_label = "Not Available"

    # Format it into a clean readable string for the agent
    # NOTE: Risk label intentionally excluded — agents must reason independently
    profile = f"""
    APPLICANT PROFILE (ID: {applicant_id})
    =======================================
    Age              : {row.get('Age', 'N/A')}
    Sex              : {row.get('Sex', 'N/A')}
    Job Type         : {job_label}
    Housing          : {row.get('Housing', 'N/A')}
    Saving Accounts  : {row.get('Saving accounts', 'N/A')}
    Checking Account : {row.get('Checking account', 'N/A')}
    Credit Amount    : {row.get('Credit amount', 'N/A')} DM
    Duration         : {row.get('Duration', 'N/A')} months
    Purpose          : {row.get('Purpose', 'N/A')}
    """
    return profile.strip(), actual_risk


# ─── Define the Data Collector Agent ─────────────────────────────────────────
data_collector_agent = Agent(
    role="Senior Financial Data Collector",

    goal=(
        "Accurately extract and present a complete, structured loan applicant "
        "profile from the BFSI dataset for further analysis."
    ),

    backstory=(
        "You are a seasoned data professional with 12 years of experience in "
        "Indian banking and financial services. You have worked with major banks "
        "like HDFC, ICICI, and SBI, extracting and structuring loan applicant "
        "data for credit assessment teams. You are meticulous, detail-oriented, "
        "and understand exactly which data points matter for credit risk decisions."
    ),

    llm=llm,
    verbose=True  # Shows the agent's thinking process in the terminal
)
