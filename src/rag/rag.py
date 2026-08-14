import pickle

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama


# --------------------------------
# 1. Load embedding model
# --------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 2. Load saved vector store
# --------------------------------

with open("vector_store.pkl", "rb") as file:
    index = pickle.load(file)

chunks = index["chunks"]
document_embeddings = index["embeddings"]


# --------------------------------
# 3. Load local LLM
# --------------------------------

llm = Llama(
    model_path="models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    verbose=False
)


# --------------------------------
# 4. Ask question
# --------------------------------

query = input("Ask a question: ")


# --------------------------------
# 5. Embed ONLY the question
# --------------------------------

query_embedding = embedding_model.encode([query])


# --------------------------------
# 6. Search vector store
# --------------------------------

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]


# --------------------------------
# 7. Rank results
# --------------------------------

results = sorted(
    zip(chunks, similarities),
    key=lambda x: x[1],
    reverse=True
)


# --------------------------------
# 8. Select Top-K
# --------------------------------
TOP_K = 3
SIMILARITY_THRESHOLD = 0.45

top_results = [
    result
    for result in results[:TOP_K]
    if result[1] >= SIMILARITY_THRESHOLD
]

# --------------------------------
# 9. Build context
# --------------------------------

context = "\n\n".join(
    result[0]["text"]
    for result in top_results
)


# --------------------------------
# 10. Build RAG prompt
# --------------------------------
prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.

Rules:
- Use only facts explicitly stated in the context.
- Do not combine unrelated facts.
- Do not make assumptions.
- Do not add outside knowledge.
- Give a short, direct answer.
- If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""

# --------------------------------
# 11. Send prompt to Qwen
# --------------------------------
response = llm(
    prompt,
    max_tokens=80,
    temperature=0.1,
    repeat_penalty=1.15,
)
# --------------------------------
# 12. Display results
# --------------------------------

print("\n" + "=" * 60)
print("TOP RETRIEVED CHUNKS")
print("=" * 60)

for chunk, score in top_results:

    print(f"\nScore: {score:.4f}")
    print(f"Document: {chunk['document']}")
    print(f"Text: {chunk['text']}")


print("\n" + "=" * 60)
print("LLM ANSWER")
print("=" * 60)

print(response["choices"][0]["text"].strip())
