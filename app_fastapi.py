# app_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
from collaborative_filtering import CollaborativeFilteringRecommender
from nlp_embeddings import EmbeddingGenerator
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch

app = FastAPI()

# Load models
cf_model = CollaborativeFilteringRecommender()
cf_model.train_model()

embedder = EmbeddingGenerator()

DIALECT_MODEL_PATH = "models/dialect_model"
dialect_tokenizer = BertTokenizerFast.from_pretrained(DIALECT_MODEL_PATH)
dialect_model = BertForSequenceClassification.from_pretrained(DIALECT_MODEL_PATH)

class TextInput(BaseModel):
    text: str

@app.get("/recommend/{user_id}")
def get_recommendations(user_id: int):
    recommendations = cf_model.get_recommendations(user_id)
    return {"user_id": user_id, "recommendations": recommendations}

@app.post("/predict_dialect")
def predict_dialect(item: TextInput):
    text = item.text
    inputs = dialect_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = dialect_model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=1).item()
    label_mapping = {0: "Standard", 1: "Informal", 2: "Formal"}
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    return {"text": text, "predicted_dialect": predicted_label}

@app.post("/get_embedding")
def get_embedding(item: TextInput):
    text = item.text
    embedding = embedder.get_embedding(text)
    return {"text": text, "embedding": embedding.tolist()}

# To run the FastAPI app use:
# uvicorn app_fastapi:app --reload
