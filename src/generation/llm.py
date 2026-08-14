from llama_cpp import Llama


MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=6,
    verbose=False,
)


response = llm.create_chat_completion(
    messages=[
        {
            "role": "user",
            "content": "What package manager does Fedora use?"
        }
    ],
    max_tokens=100,
)


answer = response["choices"][0]["message"]["content"]

print("\nLLM ANSWER:")
print(answer)
