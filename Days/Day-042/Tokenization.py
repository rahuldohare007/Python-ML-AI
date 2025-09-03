# 🚀 Day 42/100 of #100DaysOfCode
# 🎯 Tokenization, Stopwords, Lemmatization

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy

# Sample text
text = "The striped bats are hanging on their feet for best."

# 1. TOKENIZATION (NLTK)
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. REMOVING STOPWORDS
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Without Stopwords:", filtered_tokens)

# 3. LEMMATIZATION (NLTK)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatized Words:", lemmatized_words)
