from llama_cpp import Llama
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------
# 1. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
llm = Llama(
    model_path="models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    n_threads=6,
    verbose=False
)

# --------------------------------
# 2. Load documents and create chunks
# --------------------------------

documents_path = Path("data/documents")

chunks = []

for file in documents_path.glob("*.txt"):

    text = file.read_text()

    sentences = [
        sentence.strip()
        for sentence in text.replace("\n", " ").split(".")
        if sentence.strip()
    ]

    chunk_size = 2

    for i in range(0, len(sentences), chunk_size):

        chunk = ". ".join(sentences[i:i + chunk_size]) + "."

        chunks.append({
            "document": file.name,
            "text": chunk
        })


# --------------------------------
# 3. Ask the user a question
# --------------------------------

query = input("Ask a question: ")


# --------------------------------
# 4. Create embeddings
# --------------------------------

texts = [chunk["text"] for chunk in chunks]

document_embeddings = model.encode(texts)

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
# 7. Get top 2 chunks
# --------------------------------

TOP_K = 3
top_results = results[:TOP_K]


# --------------------------------
# 8. Build context
# --------------------------------

context = "\n\n".join(
    result[0]["text"]
    for result in top_results
)


# --------------------------------
# 9. Build RAG prompt
# --------------------------------

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context provided below.

Context:
{context}

Question:
{query}

Answer:
"""


# --------------------------------
# 10. Display everything
# --------------------------------

print("\n" + "=" * 60)
print("TOP RETRIEVED CHUNKS")
print("=" * 60)

for chunk, score in top_results:

    print(f"\nScore: {score:.4f}")
    print(f"Document: {chunk['document']}")
    print(f"Text: {chunk['text']}")


print("\n" + "=" * 60)
print("PROMPT SENT TO LLM")
print("=" * 60)

print(prompt)
response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer using ONLY the provided context."
        },
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=150,
    temperature=0.2
)
answer = response["choices"][0]["message"]["content"]

print("\n" + "=" * 60)
print("LLM ANSWER")
print("=" * 60)

print(answer)
