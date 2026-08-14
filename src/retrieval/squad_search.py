import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Load embedding model
# --------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 2. Connect to ChromaDB
# --------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)


# --------------------------------
# 3. Load SQuAD collection
# --------------------------------

collection = client.get_collection(
    name="rag_squad_v03"
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
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=5
)


# --------------------------------
# 7. Display results
# --------------------------------

print("\n" + "=" * 70)
print("SQUAD v0.3 CHROMADB SEARCH")
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
        f"Source: "
        f"{results['metadatas'][0][i]['source']}"
    )

    print(
        f"Text: "
        f"{results['documents'][0][i]}"
    )

print("\n" + "=" * 70)

