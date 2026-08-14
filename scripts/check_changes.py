from pathlib import Path
import pickle


# --------------------------------
# 1. Load saved index
# --------------------------------

with open("vector_store.pkl", "rb") as file:
    index = pickle.load(file)


indexed_documents = index["documents"]


# --------------------------------
# 2. Check current documents
# --------------------------------

documents_path = Path("data/documents")

for file in documents_path.glob("*.txt"):

    current_time = file.stat().st_mtime

    indexed_time = indexed_documents.get(file.name)

    if indexed_time is None:

        print(f"NEW: {file.name}")

    elif current_time != indexed_time:

        print(f"CHANGED: {file.name}")

    else:

        print(f"UNCHANGED: {file.name}")
