# 🚀 Day 47/100 of #100DaysOfCode
# 🎯 Using BERT for Classification 

from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load pre-trained BERT model & tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Create pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Test sentences
sentences = [
    "I absolutely loved this movie, it was fantastic!",
    "The product was disappointing and not worth the money.",
    "It was okay, not too good but not too bad either."
]

# Run classification
for s in sentences:
    result = classifier(s)[0]
    print(f"{s} -> {result}")
