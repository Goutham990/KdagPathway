import numpy as np
from transformers import pipeline

# Load a transformer model for embeddings
embedder = pipeline('feature-extraction', model='bert-base-uncased', tokenizer='bert-base-uncased')

# Create conference profiles (e.g., from benchmark papers)
conference_profiles = {
    "CVPR": "Computer Vision and Pattern Recognition topics.",
    "NeurIPS": "Neural Information Processing Systems topics.",
    "EMNLP": "Natural Language Processing topics.",
    # Add other conferences as needed
}

# Generate embeddings for conferences
conference_embeddings = {k: np.mean(embedder(v), axis=1) for k, v in conference_profiles.items()}

def recommend_conference(content):
    paper_embedding = np.mean(embedder(content), axis=1)
    scores = {conf: np.dot(paper_embedding, conf_emb.T) for conf, conf_emb in conference_embeddings.items()}
    best_match = max(scores, key=scores.get)
    return best_match, scores[best_match]

# Recommend conferences for publishable papers
data['conference'] = data.apply(lambda row: recommend_conference(row['content'])[0] if row['publishable'] == 1 else 'NA', axis=1)
data['rationale'] = data.apply(lambda row: f"The paper aligns with {row['conference']} focus areas." if row['publishable'] == 1 else 'NA', axis=1)