question = "What package manager does Fedora use?"

retrieved_chunks = [
    "DNF is the package manager commonly used on Fedora."
]

context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

print(prompt)
