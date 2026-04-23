import os
import requests
from crewai import Agent
from crewai.tools import tool
from agents.llm_config import llm
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_fred_series(series_id: str, units: str = "lin") -> dict:
    """
    Fetches the latest observation for a given FRED series ID.
    Returns a dict with series_id, value, date, and unit.
    """
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 1,
        "units": units,
    }
    try:
        response = requests.get(FRED_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        observations = data.get("observations", [])
        if observations:
            obs = observations[0]
            return {
                "series_id": series_id,
                "value": obs.get("value", "N/A"),
                "date": obs.get("date", "N/A"),
            }
        return {"series_id": series_id, "value": "N/A", "date": "N/A"}
    except Exception as e:
        return {"series_id": series_id, "value": f"Error: {str(e)}", "date": "N/A"}


@tool("FRED Macroeconomic Data Fetcher")
def fred_macro_tool(query: str) -> str:
    """
    Fetches key macroeconomic indicators from the FRED API:
    - UNRATE  : US Unemployment Rate (%)
    - CPIAUCSL: US Inflation Rate — Consumer Price Index (All Urban Consumers)
    - GDPC1   : US Real GDP (Billions of Chained 2017 Dollars)
    - FEDFUNDS: US Federal Funds Effective Rate (%)

    Use this tool to retrieve the latest values of these indicators
    to assess the current macroeconomic environment for credit risk.
    Input can be any string — the tool always fetches all four indicators.
    """
    indicators = {
        "Unemployment Rate (%)": fetch_fred_series("UNRATE"),
        "Inflation — CPI (Index)": fetch_fred_series("CPIAUCSL"),
        "Real GDP (Billions USD)": fetch_fred_series("GDPC1"),
        "Federal Funds Rate (%)": fetch_fred_series("FEDFUNDS"),
    }

    lines = ["MACROECONOMIC INDICATORS (Source: FRED / St. Louis Fed)\n"]
    lines.append("=" * 55)
    for label, data in indicators.items():
        lines.append(f"  {label:<30}: {data['value']}  (as of {data['date']})")
    lines.append("=" * 55)

    return "\n".join(lines)


# ─── Define the Market Analyst Agent ─────────────────────────────────────────
market_analyst_agent = Agent(
    role="BFSI Market & Economic Risk Analyst",

    goal=(
        "Fetch and interpret current macroeconomic indicators to assess "
        "the broader economic environment and its implications for credit "
        "risk in the BFSI sector."
    ),

    backstory=(
        "You are a senior macroeconomic analyst with 15 years of experience "
        "in credit risk strategy at top-tier investment banks and NBFCs. "
        "You have advised RBI-regulated institutions on how macroeconomic "
        "cycles affect loan default rates. You know that a borrower who looks "
        "creditworthy on paper can still default if the economy turns — rising "
        "unemployment and inflation are your red flags. You use live FRED data "
        "to back every assessment with real numbers."
    ),

    tools=[fred_macro_tool],
    llm=llm,
    verbose=True
)
