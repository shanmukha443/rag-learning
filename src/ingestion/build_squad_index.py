from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb


# --------------------------------
# 1. Configuration
# --------------------------------

DATASET_NAME = "rajpurkar/squad_v2"

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "rag_squad_v03"

MAX_CONTEXTS = 19029

BATCH_SIZE = 64

CHUNK_SIZE = 500

CHUNK_OVERLAP = 50


# --------------------------------
# 2. Load dataset
# --------------------------------

print("Loading SQuAD 2.0...")

dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"]


# --------------------------------
# 3. Extract unique contexts
# --------------------------------

print("Extracting unique contexts...")

unique_contexts = {}

for row in train_data:

    context = row["context"].strip()

    if context not in unique_contexts:

        unique_contexts[context] = {
            "title": row["title"],
            "source": "squad_v2",
            "split": "train",
        }


contexts = list(unique_contexts.items())


print(f"Unique contexts found: {len(contexts)}")


# --------------------------------
# 4. Limit dataset for v0.3
# --------------------------------

contexts = contexts[:MAX_CONTEXTS]

print(f"Contexts selected for v0.3: {len(contexts)}")


# --------------------------------
# 5. Load embedding model
# --------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------
# 6. Create ChromaDB client
# --------------------------------

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# --------------------------------
# 7. Create separate collection
# --------------------------------

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# --------------------------------
# 8. Create chunks
# --------------------------------

print("Creating chunks...")

documents = []
metadatas = []
ids = []


chunk_id = 0


for context, metadata in contexts:

    words = context.split()

    start = 0

    while start < len(words):

        end = start + CHUNK_SIZE

        chunk_words = words[start:end]

        chunk_text = " ".join(chunk_words).strip()

        if chunk_text:

            documents.append(chunk_text)

            metadatas.append({
                "title": metadata["title"],
                "source": metadata["source"],
                "split": metadata["split"],
            })

            ids.append(
                f"squad_v03_{chunk_id}"
            )

            chunk_id += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP


print(f"Total chunks created: {len(documents)}")


# --------------------------------
# 9. Generate embeddings in batches
# --------------------------------

print("Generating embeddings...")

for start in range(
    0,
    len(documents),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(documents)
    )

    batch_documents = documents[start:end]

    print(
        f"Embedding chunks "
        f"{start + 1}-{end} "
        f"of {len(documents)}"
    )

    embeddings = embedding_model.encode(
        batch_documents,
        batch_size=BATCH_SIZE,
        show_progress_bar=True
    )


    # --------------------------------
    # 10. Store batch in ChromaDB
    # --------------------------------

    collection.upsert(

        ids=ids[start:end],

        documents=batch_documents,

        embeddings=embeddings.tolist(),

        metadatas=metadatas[start:end]

    )


# --------------------------------
# 11. Display information
# --------------------------------

print()
print("=" * 60)
print("SQuAD v0.3 INDEX CREATED")
print("=" * 60)

print(
    f"Collection: {COLLECTION_NAME}"
)

print(
    f"Contexts: {len(contexts)}"
)

print(
    f"Chunks: {len(documents)}"
)

print(
    f"Stored chunks: {collection.count()}"
)

print("=" * 60)
