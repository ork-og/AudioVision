from llama_cpp import Llama


class TranslateAI: 
    def __init__(self):
        pass
    
    def load(self):
        self.llm = Llama.from_pretrained(
            repo_id="bartowski/gemma-2-9b-it-GGUF",
            filename="gemma-2-9b-it-Q4_K_M.gguf",
            chat_format="gemma",
        )

    def translatePromt(self, txt: str):
        messages = [
            {"role": "user", "content": f"Translate into English: {txt}"}
        ]
        tokens = int(len(txt.split())*1.25 + 5)
        out = self.llm.create_chat_completion(messages=messages, max_tokens=tokens)
        return out["choices"][0]["message"]["content"] 
