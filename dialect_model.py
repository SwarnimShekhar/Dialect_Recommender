# dialect_model.py
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

MODEL_SAVE_PATH = "models/dialect_model"

class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_dialect_model(data_file='data/dialect_data.csv'):
    # Load the dataset
    df = pd.read_csv(data_file)
    texts = df['text'].tolist()
    # Create a mapping from label to integer
    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    labels = [label_mapping[label] for label in df['label']]
    
    # Load the pre-trained tokenizer and model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

    dataset = DialectDataset(texts, labels, tokenizer)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=5,
        save_total_limit=1,
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print("Dialect model fine-tuned and saved to", MODEL_SAVE_PATH)

if __name__ == '__main__':
    train_dialect_model()
