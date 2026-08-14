from pathlib import Path


text = Path("data/documents/linux.txt").read_text()

# Split document into sentences
sentences = [
    sentence.strip()
    for sentence in text.replace("\n", " ").split(".")
    if sentence.strip()
]

print("SENTENCES")
print("=" * 50)

for i, sentence in enumerate(sentences, start=1):
    print(f"{i}: {sentence}.")


# Create chunks containing 2 sentences
chunk_size = 2

chunks = []

for i in range(0, len(sentences), chunk_size):
    chunk = ". ".join(sentences[i:i + chunk_size]) + "."
    chunks.append(chunk)


print("\n")
print("CHUNKS")
print("=" * 50)

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk)
