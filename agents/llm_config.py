import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

# ─── Shared LLM — used by ALL agents in this project ─────────────────────────
# All 6 agents use the same LLM instance (Groq - Free tier)
# When deploying to Azure, we will swap this to Azure OpenAI GPT-4o
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    max_tokens=1024,
    max_retries=6           # Auto-retry on rate limit with exponential backoff
)
