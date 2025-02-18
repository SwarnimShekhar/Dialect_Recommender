# nlp_embeddings.py
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

if __name__ == '__main__':
    embedder = EmbeddingGenerator()
    text = "Hello, how are you?"
    embedding = embedder.get_embedding(text)
    print(f"Embedding for '{text}':\n", embedding)
