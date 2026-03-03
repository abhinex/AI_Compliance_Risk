from sentence_transformers import SentenceTransformer

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("BAAI/bge-large-en")
    return _model

def embed_documents(texts):
    model = get_model()
    texts = [
        "Represent this sentence for retrieval: " + t
        for t in texts
    ]
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

def embed_query(query):
    model = get_model()
    query = "Represent this question for retrieving supporting documents: " + query
    return model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )