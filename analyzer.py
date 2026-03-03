import os
import json
from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------
# Environment Setup
# ---------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
MIN_SIMILARITY_THRESHOLD = 0.30  # Prevent weak hallucinated matches

if not API_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables.")

client = Groq(api_key=API_KEY)

# ---------------------------------------------------
# System Prompt
# ---------------------------------------------------

SYSTEM_PROMPT = """
You are a senior legal contract analyst.

STRICT RULES:
- Use ONLY the retrieved context.
- Do NOT assume anything not written.
- If clause is not found → status = "missing"
- If clause exists but has blanks/TBD → status = "present_but_incomplete"
- Otherwise → status = "present"

Return STRICT JSON only:

{
  "status": "present | missing | present_but_incomplete",
  "risk_level": "low | medium | high",
  "analysis": "Detailed reasoning strictly based on retrieved text",
  "suggested_revision": "Concrete improvement suggestion",
  "confidence": 0-100
}
"""

# ---------------------------------------------------
# Main Function
# ---------------------------------------------------

def analyze_clause_with_llm(query, retrieved_results):

    # -----------------------------
    # No retrieval case
    # -----------------------------
    if not retrieved_results:
        return {
            "status": "missing",
            "risk_level": "high",
            "analysis": "No relevant clause retrieved from vector search.",
            "suggested_revision": f"Insert a properly drafted '{query}' clause.",
            "confidence": 95
        }

    # -----------------------------
    # Similarity Threshold Check
    # -----------------------------
    top_score = retrieved_results[0][0]

    if top_score < MIN_SIMILARITY_THRESHOLD:
        return {
            "status": "missing",
            "risk_level": "high",
            "analysis": "No sufficiently relevant clause found based on similarity threshold.",
            "suggested_revision": f"Insert a properly drafted '{query}' clause.",
            "confidence": 90
        }

    # -----------------------------
    # Build Context (Top 3 Chunks)
    # -----------------------------
    context_blocks = []

    for score, chunk in retrieved_results[:3]:
        context_blocks.append(
            f"Clause Heading: {chunk.get('clause_id', 'N/A')}\n"
            f"Similarity Score: {round(score, 3)}\n"
            f"Clause Text:\n{chunk.get('clause_text', '')}\n"
        )

    context = "\n\n".join(context_blocks)

    user_prompt = f"""
Query:
{query}

Retrieved Context:
{context}

Analyze strictly using only the retrieved context.
Return JSON only.
"""

    # -----------------------------
    # LLM Call
    # -----------------------------
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=800
        )

        content = response.choices[0].message.content.strip()

    except Exception as e:
        return {
            "status": "error",
            "risk_level": "unknown",
            "analysis": f"LLM API call failed: {str(e)}",
            "suggested_revision": "Check LLM service configuration.",
            "confidence": 0
        }

    # -----------------------------
    # JSON Cleanup & Parsing
    # -----------------------------
    try:
        # Remove markdown JSON fencing if present
        if content.startswith("```"):
            content = content.split("```")[1].strip()

        parsed = json.loads(content)

        return parsed

    except json.JSONDecodeError:
        return {
            "status": "error",
            "risk_level": "unknown",
            "analysis": content,
            "suggested_revision": "Manual review required due to invalid JSON output from LLM.",
            "confidence": 0
        }