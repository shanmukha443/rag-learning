from pathlib import Path

documents_path = Path("data/documents")

for file in documents_path.glob("*.txt"):
    print("=" * 50)
    print("FILE:", file.name)
    print("=" * 50)

    text = file.read_text()

    print(text)
    print()
