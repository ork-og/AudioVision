from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="DevQuasar/google.gemma-3-12b-pt-GGUF",
	filename="google.gemma-3-12b-pt.Q2_K.gguf",
)

output = llm(
	"Once upon a time, переведи этот тест с английского на русский",
	max_tokens=512,
	echo=True
)
print(output)