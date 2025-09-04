# 🚀 Day 43/100 of #100DaysOfCode
# 🎯 Text Vectorization (BoW & TF-IDF) 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample corpus
corpus = [
    "Natural Language Processing makes machines understand text.",
    "Text preprocessing is essential in NLP.",
    "TF-IDF and Bag of Words are common vectorization techniques."
]

# 1. Bag of Words (Count)
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(corpus)

print("🔹 Bag of Words Vocabulary:\n", bow_vectorizer.vocabulary_)
print("\n🔹 BoW Matrix (dense):\n", bow_matrix.toarray())

# 2. TF-IDF (Term Frequency - Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print("\n🔹 TF-IDF Vocabulary:\n", tfidf_vectorizer.vocabulary_)
print("\n🔹 TF-IDF Matrix (dense):\n", tfidf_matrix.toarray())

# Shape check
print("\nBoW Shape:", bow_matrix.shape)
print("TF-IDF Shape:", tfidf_matrix.shape)
