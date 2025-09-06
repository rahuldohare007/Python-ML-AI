# 🚀 Day 45/100 of #100DaysOfCode
# 🎯 Word Embeddings (Word2Vec, GloVe)

import os
import gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Download tokenizer
nltk.download("punkt", quiet=True)

# -------------------------------
# Sample corpus
corpus = [
    "I love natural language processing",
    "Word embeddings capture semantic meaning",
    "Word2Vec and GloVe are powerful techniques",
    "Deep learning improves NLP applications",
]

# Tokenization
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# -------------------------------
# Train Word2Vec model (Skip-gram)
w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    sg=1,            # 1 = skip-gram, 0 = CBOW
    min_count=1,
    workers=4
)

# Explore embeddings
print("\nSimilar words to 'language':", w2v_model.wv.most_similar("language"))
print("Vector for 'nlp':\n", w2v_model.wv["nlp"][:10])  # first 10 dims

# -------------------------------
# Load Pre-trained GloVe Embeddings
def load_glove(file_path):
    embeddings = {}
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = list(map(float, values[1:]))
            embeddings[word] = vector
    return embeddings

glove_path = "glove.6B.50d.txt"

if os.path.exists(glove_path):
    print("\n✅ Loading GloVe embeddings...")
    glove_embeddings = load_glove(glove_path)
    print("Vector for 'king':", glove_embeddings["king"][:10])  # first 10 dims
    print("Vector for 'queen':", glove_embeddings["queen"][:10])
else:
    print("\n⚠️ GloVe file not found. Please place 'glove.6B.50d.txt' in the same directory.")
