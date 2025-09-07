# 🚀 Day 46/100 of #100DaysOfCode
# 🎯 Intro to Transformers & HuggingFace

from transformers import pipeline

# Force use of PyTorch instead of TensorFlow
sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")

# Load a pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Test sentences
sentences = [
    "I absolutely loved the movie! It was fantastic",
    "The food was terrible and I will never come back.",
]

print("Sentiment Analysis:")
for text in sentences:
    result = sentiment_analyzer(text)[0]
    print(f"Text: {text}")
    print(f"Prediction: {result['label']} (score: {result['score']:.2f})\n")

# Zero-shot classification
classifier = pipeline("zero-shot-classification")

text = "I want to learn Machine Learning and Deep Learning."
labels = ["Education", "Sports", "Politics"]

print("Zero-shot Classification:")
result = classifier(text, candidate_labels=labels)
print(result)

# Text Generation (GPT-2)
generator = pipeline("text-generation", model="gpt2")

print("Text Generation:")
print(generator("Artificial Intelligence is transforming", max_length=30, num_return_sequences=1)[0]["generated_text"])
