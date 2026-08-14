from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer


# --------------------------------
# 1. Paths
# --------------------------------

documents_path = Path("data/documents")
vector_store_path = Path("vector_store.pkl")


# --------------------------------
# 2. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 3. Load documents and create chunks
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
# 4. Create embeddings
# --------------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)


# --------------------------------
# 5. Save index
# --------------------------------

index = {
    "chunks": chunks,
    "embeddings": embeddings,
    "documents": {
        file.name: file.stat().st_mtime
        for file in documents_path.glob("*.txt")
    }
}


with open(vector_store_path, "wb") as file:
    pickle.dump(index, file)


# --------------------------------
# 6. Display information
# --------------------------------

print("Index created successfully!")
print(f"Total chunks: {len(chunks)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Saved to: {vector_store_path}")
