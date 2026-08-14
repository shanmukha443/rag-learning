from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------
# 1. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 2. Load saved vector store
# --------------------------------

with open("vector_store.pkl", "rb") as file:
    index = pickle.load(file)

chunks = index["chunks"]
document_embeddings = index["embeddings"]


print(f"Loaded {len(chunks)} chunks.")
print(f"Embedding dimensions: {document_embeddings.shape[1]}")


# --------------------------------
# 3. Ask question
# --------------------------------

query = input("\nAsk a question: ")


# --------------------------------
# 4. Embed ONLY the question
# --------------------------------

query_embedding = model.encode([query])


# --------------------------------
# 5. Calculate similarity
# --------------------------------

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]


# --------------------------------
# 6. Rank results
# --------------------------------

results = sorted(
    zip(chunks, similarities),
    key=lambda x: x[1],
    reverse=True
)


# --------------------------------
# 7. Get Top-K
# --------------------------------

TOP_K = 2

top_results = results[:TOP_K]


# --------------------------------
# 8. Display results
# --------------------------------

print("\n" + "=" * 60)
print("TOP RETRIEVED CHUNKS")
print("=" * 60)

for chunk, score in top_results:

    print(f"\nScore: {score:.4f}")
    print(f"Document: {chunk['document']}")
    print(f"Text: {chunk['text']}")
