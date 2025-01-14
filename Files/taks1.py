import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline
import pathway as pw
 # Load the dataset
data = pd.read_csv('research_papers.csv')  # Replace with your dataset file path
labeled_data = pd.read_csv('labeled_papers.csv')  # Labeled examples (15 papers)

# Extract features and labels for Task 1
X = labeled_data['content']  # Research paper content
y = labeled_data['publishable']  # 1 for Publishable, 0 for Non-Publishable
\
 # TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tfidf, y)
def classify_paper(content):
    features = tfidf.transform([content])
    prediction = clf.predict(features)
    return prediction[0]

data['publishable'] = data['content'].apply(classify_paper)