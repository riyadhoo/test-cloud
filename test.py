import pandas as pd


# 1. Load your dataset
df = pd.read_csv('emails_dataset.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]

model = MultinomialNB()
model.fit(X, y)