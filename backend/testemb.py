from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:0.6b",
    base_url="http://localhost:11434",
)

# This will load the model and use VRAM
vector = embeddings.embed_query("Hello world")