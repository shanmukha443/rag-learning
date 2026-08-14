from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Paths
# --------------------------------

documents_path = Path("data/documents")


# --------------------------------
# 2. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 3. Create ChromaDB client
# --------------------------------

client = chromadb.PersistentClient(
    path="chroma_db"
)


# --------------------------------
# 4. Create collection
# --------------------------------

collection = client.get_or_create_collection(
    name="rag_documents"
)


# --------------------------------
# 5. Read documents and create chunks
# --------------------------------

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

        chunk = ". ".join(
            sentences[i:i + chunk_size]
        ) + "."

        chunks.append({
            "document": file.name,
            "text": chunk
        })


# --------------------------------
# 6. Create embeddings
# --------------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)


# --------------------------------
# 7. Store in ChromaDB
# --------------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

metadatas = [
    {
        "document": chunk["document"]
    }
    for chunk in chunks
]


collection.upsert(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# --------------------------------
# 8. Display information
# --------------------------------

print("ChromaDB index created successfully!")
print(f"Total chunks: {len(chunks)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Collection: {collection.name}")
