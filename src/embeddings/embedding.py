from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "package management",
    "DNF is a package manager used in Fedora.",
    "Git is used for version control."
]

embeddings = model.encode(texts)

similarity = cosine_similarity(embeddings)

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        print()
        print(f"Text A: {texts[i]}")
        print(f"Text B: {texts[j]}")
        print(f"Similarity: {similarity[i][j]:.4f}")