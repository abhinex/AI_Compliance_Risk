import os
import json
import sys

from ingestion.pdf_extractor import extract_text_from_pdf
from chunking.clause_chunker import chunk_clauses
from embeddings.embedder import embed_documents, embed_query
from vectorstore.faiss_store import FaissStore
from reasoning.analyzer import analyze_clause_with_llm


# ===================================================
# CONFIG
# ===================================================

PDF_PATH = "/Users/Abhishek/Desktop/Projects/Document_Intelligentce_system/BE/data/AI_Test_Compliant_Contract.pdf"
QUERY = "What is the termination clause?"
TOP_K = 8


# ===================================================
# 1️⃣ Extract
# ===================================================

print("\n🔹 STEP 1: Extracting PDF...")

pages = extract_text_from_pdf(PDF_PATH)

if not pages:
    print("❌ No pages extracted. Exiting.")
    sys.exit()

print(f"✅ Extracted {len(pages)} pages.")


# ===================================================
# 2️⃣ Chunk
# ===================================================

print("\n🔹 STEP 2: Chunking clauses...")

chunks = chunk_clauses(pages)

if not chunks:
    print("❌ No clauses detected. Exiting.")
    sys.exit()

print(f"✅ Total clauses detected: {len(chunks)}")


# ===================================================
# 3️⃣ Embed Documents
# ===================================================

print("\n🔹 STEP 3: Embedding documents...")

texts = [chunk["clause_text"] for chunk in chunks]

embeddings = embed_documents(texts)

if embeddings is None or len(embeddings) == 0:
    print("❌ Embeddings failed. Exiting.")
    sys.exit()

print(f"✅ Generated embeddings shape: {embeddings.shape}")


# ===================================================
# 4️⃣ Build FAISS Index
# ===================================================

print("\n🔹 STEP 4: Building FAISS index...")

dim = embeddings.shape[1]
store = FaissStore(dim)
store.add(embeddings, chunks)

print("✅ FAISS index built successfully.")


# ===================================================
# 5️⃣ Query
# ===================================================

print("\n🔹 STEP 5: Processing Query")
print(f"Query: {QUERY}")

query_embedding = embed_query(QUERY)

if query_embedding is None:
    print("❌ Query embedding failed.")
    sys.exit()

results = store.search(query_embedding, k=TOP_K)

if not results:
    print("❌ No retrieval results found.")
    sys.exit()


# ===================================================
# 6️⃣ Boost Heading Match
# ===================================================

query_lower = QUERY.lower()

def boost_score(result):
    score, chunk = result
    boost = 0

    if any(word in chunk["clause_id"].lower() for word in query_lower.split()):
        boost += 0.1

    return score + boost

results = sorted(results, key=boost_score, reverse=True)


# ===================================================
# 7️⃣ Show Retrieval Results
# ===================================================

print("\n🔹 STEP 6: Top Retrieval Results\n")

for rank, (score, chunk) in enumerate(results[:5], start=1):
    print(f"Rank {rank}")
    print(f"Score: {round(score, 4)}")
    print(f"Heading: {chunk['clause_id']}")
    print("-" * 60)


# ===================================================
# 8️⃣ LLM Analysis
# ===================================================

print("\n🔹 STEP 7: Running LLM Analysis...\n")

analysis = analyze_clause_with_llm(QUERY, results)

print("🔹 LLM ANALYSIS OUTPUT:\n")
print(json.dumps(analysis, indent=2))


# ===================================================
# 9️⃣ Confidence Warning
# ===================================================

confidence = analysis.get("confidence", 0)

if confidence < 60:
    print("\n⚠️  WARNING: Low confidence result. Manual review recommended.")


print("\n✅ Pipeline Completed Successfully.\n")