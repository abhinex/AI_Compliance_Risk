import faiss
import numpy as np

class FaissStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)  # use cosine similarity
        self.metadata_store = []

    def add(self, embeddings, metadata_list):
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        self.index.add(embeddings)
        self.metadata_store.extend(metadata_list)

    def search(self, query_embedding, k=5):
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append((score, self.metadata_store[idx]))

        return results

       