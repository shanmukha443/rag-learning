from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Find our documents
documents_path = Path("data/documents")

documents = []
texts = []

for file in documents_path.glob("*.txt"):
    text = file.read_text()

    documents.append(file.name)
    texts.append(text)


# Ask the user a question
query = input("Ask a question: ")


# Convert documents and query into vectors
document_embeddings = model.encode(texts)
query_embedding = model.encode([query])


# Calculate similarity
similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]


# Display results
results = sorted(
    zip(documents, similarities),
    key=lambda x: x[1],
    reverse=True
)


print("\nSearch results:\n")

for document, score in results:
    print(f"{document}: {score:.4f}")
