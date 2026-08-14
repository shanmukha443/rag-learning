import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 2. Connect to existing ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)


# --------------------------------
# 3. Get collection
# --------------------------------

collection = client.get_collection(
    name="rag_documents"
)


# --------------------------------
# 4. Ask question
# --------------------------------

query = input("Ask a question: ")


# --------------------------------
# 5. Create query embedding
# --------------------------------

query_embedding = model.encode(
    [query]
)[0]


# --------------------------------
# 6. Search ChromaDB
# --------------------------------

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)
# --------------------------------
# 7. Display results
# --------------------------------

print("\n" + "=" * 60)
print("CHROMADB SEARCH RESULTS")
print("=" * 60)

for i in range(len(results["documents"][0])):

    print(f"\nResult {i + 1}")

    print(
        f"Document: "
        f"{results['metadatas'][0][i]['document']}"
    )

    print(
        f"Text: "
        f"{results['documents'][0][i]}"
    )

    print(
        f"Distance: "
        f"{results['distances'][0][i]:.4f}"
    )
