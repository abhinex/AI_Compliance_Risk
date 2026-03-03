import re

CLAUSE_PATTERNS = [
    r"^\s*\d+(\.\d+)*\s+[A-Z].*",
    r"^\s*ARTICLE\s+\d+[A-Z]?\b.*",
    r"^\s*SECTION\s+\d+(\.\d+)*\b.*",
    r"^\s*§\s*\d+(\.\d+)*\s+.*"
]

def chunk_clauses(pages):
    full_text = "\n".join([p["text"] for p in pages])

    combined_pattern = "|".join(CLAUSE_PATTERNS)

    matches = list(
        re.finditer(combined_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    )

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(full_text)

        clause_text = full_text[start:end].strip()

        if len(clause_text) > 150:
            chunks.append({
                "clause_id": match.group().strip(),
                "clause_text": clause_text
            })

    return chunks