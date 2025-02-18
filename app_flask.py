# app_flask.py
from flask import Flask, request, jsonify, send_from_directory
from collaborative_filtering import CollaborativeFilteringRecommender
from nlp_embeddings import EmbeddingGenerator
from transformers import BertForSequenceClassification, BertTokenizerFast
import torch, os
import numpy as np

app = Flask(__name__)

# Load the collaborative filtering model
cf_model = CollaborativeFilteringRecommender()
cf_model.train_model()

# Load the NLP embedding model
embedder = EmbeddingGenerator()

# Load the fine-tuned dialect recognition model
DIALECT_MODEL_PATH = "models/dialect_model"
dialect_tokenizer = BertTokenizerFast.from_pretrained(DIALECT_MODEL_PATH)
dialect_model = BertForSequenceClassification.from_pretrained(DIALECT_MODEL_PATH)


@app.route('/')
def home():
    return 'Welcome to our website!'

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )
    
@app.route('/recommend', methods=['GET'])
def recommend():
    try:
        user_id = int(request.args.get('user_id'))
    except (ValueError, TypeError):
        return jsonify({"error": "Please provide a valid user_id"}), 400
    recommendations = cf_model.get_recommendations(user_id)
    return jsonify({"user_id": user_id, "recommendations": recommendations})

@app.route('/predict_dialect', methods=['POST'])
def predict_dialect():
    data = request.get_json()
    text = data.get("text", "")
    inputs = dialect_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = dialect_model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=1).item()
    # Mapping: (This should match your training labels)
    label_mapping = {0: "Standard", 1: "Informal", 2: "Formal"}
    predicted_label = label_mapping.get(predicted_class, "Unknown")
    return jsonify({"text": text, "predicted_dialect": predicted_label})

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    data = request.get_json()
    text = data.get("text", "")
    embedding = embedder.get_embedding(text)
    return jsonify({"text": text, "embedding": embedding.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
