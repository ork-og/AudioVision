from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="bartowski/gemma-2-9b-it-GGUF",
    filename="gemma-2-9b-it-Q4_K_M.gguf",
    chat_format="gemma",
)

messages = [
    {"role": "user", "content": "Translate into English: Девушка в наушниках сидит у окна в вечернем городе, мягкий тёплый свет лампы, уютная комната, атмосферный Lo-Fi стиль, неоновые отблески, дождь за окном, расслабленная атмосфера, тёплые цвета, мягкое рассеянное освещение, стиль аниме"}
]

out = llm.create_chat_completion(messages=messages, max_tokens=100)
print(out["choices"][0]["message"]["content"])
