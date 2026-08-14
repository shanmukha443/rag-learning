from pathlib import Path


DOCUMENTS_DIR = Path("data/documents")


def load_documents():
    documents = {}

    for file in DOCUMENTS_DIR.glob("*.txt"):
        documents[file.name] = file.read_text()

    return documents


def search_documents(query, documents):
    results = []

    for filename, content in documents.items():
        if query.lower() in content.lower():
            results.append(filename)

    return results


documents = load_documents()

query = input("Ask a question: ")

results = search_documents(query, documents)

print("\nRelevant documents:")

for filename in results:
    print(f"- {filename}")