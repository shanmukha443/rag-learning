import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama


# --------------------------------
# 1. Configuration
# --------------------------------

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "rag_squad_v03"

MODEL_PATH = (
    "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

TOP_K = 1


# --------------------------------
# 2. Load embedding model
# --------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 3. Connect to ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# --------------------------------
# 4. Load local LLM
# --------------------------------

print("Loading local Qwen model...")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    verbose=False
)


# --------------------------------
# 5. Ask question
# --------------------------------

query = input("\nAsk a question: ")


# --------------------------------
# 6. Embed question
# --------------------------------

query_embedding = embedding_model.encode(
    [query]
)[0]


# --------------------------------
# 7. Retrieve relevant chunks
# --------------------------------

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=TOP_K
)


# --------------------------------
# 8. Build context
# --------------------------------

retrieved_chunks = []

for i in range(
    len(results["documents"][0])
):

    text = results["documents"][0][i]

    title = results["metadatas"][0][i]["title"]

    retrieved_chunks.append(
        f"Source: {title}\n"
        f"Context: {text}"
    )


context = "\n\n".join(
    retrieved_chunks
)


# --------------------------------
# 9. Build RAG prompt
# --------------------------------

prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{query}

The answer is:
"""

# --------------------------------
# 10. Generate answer
# --------------------------------

response = llm(
    prompt,
    max_tokens=10,
    temperature=0.0,
    repeat_penalty=1.1,
)

# --------------------------------
# 11. Display retrieved sources
# --------------------------------

print("\n" + "=" * 70)
print("TOP RETRIEVED CONTEXT")
print("=" * 70)

for i in range(
    len(results["documents"][0])
):

    print(
        f"\nResult {i + 1}"
    )

    print(
        f"Distance: "
        f"{results['distances'][0][i]:.4f}"
    )

    print(
        f"Title: "
        f"{results['metadatas'][0][i]['title']}"
    )

    print(
        f"Text: "
        f"{results['documents'][0][i]}"
    )


# --------------------------------
# 12. Display final answer
# --------------------------------

print("\n" + "=" * 70)
print("QWEN RAG ANSWER")
print("=" * 70)

answer = response["choices"][0]["text"].strip()

# Keep only the first non-empty line
answer = answer.splitlines()[0].strip()

print(answer)
# --------------------------------
# 13. Clean up LLM
# --------------------------------

llm.close()
