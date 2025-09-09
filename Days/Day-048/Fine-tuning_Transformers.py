# 🚀 Day 48/100 of #100DaysOfCode
# 🎯 Fine-tuning Transformers on custom text 

import pandas as pd
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# 1. Load custom dataset (CSV with 'text', 'label' columns)
df = pd.read_csv("custom_dataset.csv")  # Example: reviews + sentiment labels
dataset = Dataset.from_pandas(df)

# 2. Load tokenizer
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 3. Train-test split
dataset = dataset.train_test_split(test_size=0.2)
train_ds, test_ds = dataset["train"], dataset["test"]

# 4. Load model
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(set(df["label"])))

# 5. Training setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
)

# 6. Train
trainer.train()

# 7. Evaluate
print(trainer.evaluate())
