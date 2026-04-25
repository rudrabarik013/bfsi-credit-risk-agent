import os
import numpy as np
from crewai import Agent
from crewai.tools import tool
from agents.llm_config import llm
from sentence_transformers import SentenceTransformer
import faiss

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(BASE_DIR, "docs", "rbi_credit_compliance_guidelines.txt")


# ─── Step 1: Load & Chunk the RBI Document ───────────────────────────────────
def load_and_chunk_document(file_path: str) -> list:
    """
    Reads the compliance document and splits it into meaningful chunks.
    Splits on double newlines (paragraphs/sections) for semantic coherence.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 40]

    # Group small paragraphs into ~1200-character chunks
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < 1200:
            current_chunk += "\n\n" + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ─── Step 2: Build FAISS Vector Index at module load time ────────────────────
print("  [RAG] Loading embedding model (sentence-transformers)...")
_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("  [RAG] Chunking RBI compliance document...")
_chunks = load_and_chunk_document(DOC_PATH)

print(f"  [RAG] Embedding {len(_chunks)} chunks...")
_embeddings = _embedding_model.encode(_chunks, convert_to_numpy=True)

# Normalize vectors → cosine similarity via inner product
_norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
_embeddings_normalized = (_embeddings / _norms).astype(np.float32)

# Build FAISS flat index
_faiss_index = faiss.IndexFlatIP(_embeddings_normalized.shape[1])
_faiss_index.add(_embeddings_normalized)

print(f"  [RAG] Index ready — {_faiss_index.ntotal} chunks indexed. ✅\n")


# ─── Step 3: Python Hard Compliance Pre-Checker ──────────────────────────────
# Deterministic checks that the LLM keeps getting wrong — handled in Python
# Same pattern as Agent 2's hard_validation_tool

SAVINGS_CREDIT_LIMITS = {
    "little":      5000,
    "moderate":    15000,
    "quite rich":  30000,
    "rich":        50000,
}

PURPOSE_DURATION_LIMITS = {
    "radio/tv":             36,
    "furniture":            36,
    "car":                  60,
    "domestic appliances":  24,
    "repairs":              48,
    "education":            84,
    "business":             48,
    "vacation":             12,
    "retraining":           36,
}


@tool("Python Hard Compliance Checker")
def python_compliance_tool(applicant_profile: str) -> str:
    """
    Runs deterministic Python-based compliance checks on an applicant profile.
    Handles checks that require precise arithmetic or policy statements —
    things the LLM consistently gets wrong.

    Checks performed:
    1. Age >= 21 (minimum eligibility)
    2. Credit amount vs savings category limit
    3. Loan duration vs purpose-specific maximum
    4. Gender non-discrimination (policy statement — always COMPLIANT)

    Input : Full applicant profile text.
    Output: PASS/FAIL for each check with exact rule cited — 100% deterministic.
    """
    lines = applicant_profile.strip().splitlines()
    parsed = {}
    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            parsed[key.strip()] = value.strip()

    results = ["PYTHON HARD COMPLIANCE CHECK (100% Deterministic)"]
    results.append("=" * 60)

    # ── CHECK 1: Age ───────────────────────────────────────────────────────────
    results.append("\n[CHECK 1] AGE ELIGIBILITY")
    try:
        age = float(parsed.get("Age", "0"))
        if age >= 21:
            results.append(f"  ✅ COMPLIANT — Age {int(age)} >= 21 (RBI minimum). No violation.")
        else:
            results.append(f"  ❌ VIOLATION — Age {int(age)} < 21. Borrowers below 21 are ineligible.")
    except ValueError:
        results.append("  ⚠️  UNKNOWN — Age value could not be parsed.")

    # ── CHECK 2: Credit Amount vs Savings Limit ────────────────────────────────
    results.append("\n[CHECK 2] CREDIT AMOUNT vs SAVINGS THRESHOLD")
    try:
        credit_raw = parsed.get("Credit Amount", "0").replace("DM", "").strip()
        credit = float(credit_raw)
        savings = parsed.get("Saving Accounts", "").strip().lower()
        limit = SAVINGS_CREDIT_LIMITS.get(savings)
        if limit is None:
            results.append(f"  ⚠️  UNKNOWN — Savings category '{savings}' not in limit table.")
        elif credit > limit:
            results.append(
                f"  ❌ VIOLATION — Credit Amount {credit:.0f} DM exceeds "
                f"the {limit} DM limit for '{savings}' savings category. "
                f"(RBI Guideline: Borrowers with '{savings}' savings: credit must not exceed {limit} DM)"
            )
        else:
            results.append(
                f"  ✅ COMPLIANT — Credit Amount {credit:.0f} DM is within "
                f"the {limit} DM limit for '{savings}' savings category."
            )
    except ValueError:
        results.append("  ⚠️  UNKNOWN — Credit amount could not be parsed.")

    # ── CHECK 3: Duration vs Purpose Limit ────────────────────────────────────
    results.append("\n[CHECK 3] LOAN DURATION vs PURPOSE MAXIMUM")
    try:
        duration_raw = parsed.get("Duration", "0").replace("months", "").strip()
        duration = float(duration_raw)
        purpose = parsed.get("Purpose", "").strip().lower()
        limit = PURPOSE_DURATION_LIMITS.get(purpose)
        if limit is None:
            results.append(f"  ⚠️  UNKNOWN — Purpose '{purpose}' not in duration limit table.")
        elif duration > limit:
            results.append(
                f"  ❌ VIOLATION — Duration {int(duration)} months exceeds "
                f"the {limit}-month maximum for '{purpose}' loans. "
                f"(RBI Guideline: Max duration for {purpose}: {limit} months)"
            )
        else:
            results.append(
                f"  ✅ COMPLIANT — Duration {int(duration)} months is within "
                f"the {limit}-month maximum for '{purpose}' loans."
            )
    except ValueError:
        results.append("  ⚠️  UNKNOWN — Duration could not be parsed.")

    # ── CHECK 4: Gender Non-Discrimination ────────────────────────────────────
    results.append("\n[CHECK 4] GENDER NON-DISCRIMINATION POLICY")
    results.append(
        "  ✅ COMPLIANT — Recording gender as applicant data is permitted. "
        "RBI non-discrimination policy prohibits using gender as a RISK FACTOR "
        "in credit decisions. Our system does not use gender as a risk factor."
    )

    results.append("\n" + "=" * 60)
    results.append("Python hard compliance check complete.")
    return "\n".join(results)


# ─── Step 4: RAG Retrieval Tool ──────────────────────────────────────────────
@tool("RBI Compliance Rules Retriever")
def rbi_compliance_tool(query: str) -> str:
    """
    Searches the RBI Credit Compliance Guidelines using semantic similarity (RAG).

    Input : A natural language query about a specific compliance rule.
            Examples:
              - "age eligibility minimum maximum loan"
              - "credit amount limit for little savings account"
              - "loan duration radio TV consumer goods maximum months"
              - "unskilled borrower credit limit collateral"
              - "gender sex discrimination lending non-discrimination"

    Output: The top 3 most relevant sections from the RBI compliance document,
            ranked by semantic similarity score.

    Call this tool MULTIPLE TIMES with different queries to check each
    compliance aspect of the applicant profile independently.
    """
    # Embed and normalize the query
    query_vec = _embedding_model.encode([query], convert_to_numpy=True)
    query_vec = (query_vec / np.linalg.norm(query_vec)).astype(np.float32)

    # Retrieve top 1 most relevant chunk to stay within Groq token limits
    scores, indices = _faiss_index.search(query_vec, k=1)

    lines = [f"RBI COMPLIANCE RULES — Query: '{query}'", "=" * 60]
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk_text = _chunks[idx][:800]  # Truncate to 800 chars to control tokens
        lines.append(f"\n[Result {rank + 1} | Similarity Score: {score:.3f}]")
        lines.append(chunk_text)
    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


# ─── Step 4: Compliance Agent ─────────────────────────────────────────────────
compliance_agent = Agent(
    role="RBI Regulatory Compliance Officer",

    goal=(
        "Check the loan applicant's profile against RBI credit compliance "
        "guidelines using the RAG retrieval tool. Identify all regulatory "
        "violations, flag non-compliant fields, and produce a structured "
        "compliance report with a final COMPLIANT / NON-COMPLIANT verdict."
    ),

    backstory=(
        "You are a senior RBI-certified compliance officer with 18 years of "
        "experience in regulatory oversight at scheduled commercial banks. "
        "You have conducted thousands of credit file audits and know the RBI "
        "Fair Practices Code for Lenders inside out. You are meticulous and "
        "uncompromising — a single regulatory violation is enough to halt a "
        "loan sanction. You use a RAG-based retrieval system to fetch the "
        "exact RBI rule applicable to each field before making any compliance "
        "judgement. You never rely on memory — you always retrieve first."
    ),

    tools=[python_compliance_tool, rbi_compliance_tool],
    llm=llm,
    verbose=True
)
