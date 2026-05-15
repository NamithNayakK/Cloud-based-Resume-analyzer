"""
Create Sample Resume Classifier Model
Generates a simple Naive Bayes model for testing without external CSV data.
"""

import os
import pickle
import math
import re
from collections import Counter

class SimpleNaiveBayesClassifier:
    """Lightweight Naive Bayes text classifier for resume categorization."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
        "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them",
        "this", "to", "was", "were", "with", "will", "would", "can", "could", "should", "may",
    }

    def __init__(self):
        self.class_freq = {}
        self.word_freq = {}
        self.categories = set()
        self.vocab = set()
        self.total_words = 0
        
    def tokenize(self, text):
        """Convert text to lowercase tokens."""
        if not text:
            return []
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = []
        for word in text.split():
            word = self.normalize_token(word)
            if len(word) > 2 and word not in self.STOP_WORDS:
                tokens.append(word)
        return tokens

    def normalize_token(self, token):
        """Apply light stemming."""
        token = token.strip()
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        for suffix in ("ingly", "edly", "ing", "ed", "ers", "er", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[:-len(suffix)]
        return token
    
    def train(self, X, y):
        """Train classifier on text and labels."""
        print(f"[INFO] Training on {len(X)} samples...")
        
        for text, category in zip(X, y):
            self.categories.add(category)
            self.class_freq[category] = self.class_freq.get(category, 0) + 1
            
            if category not in self.word_freq:
                self.word_freq[category] = {}
            
            tokens = self.tokenize(text)
            for token in tokens:
                self.vocab.add(token)
                self.word_freq[category][token] = self.word_freq[category].get(token, 0) + 1
                self.total_words += 1
        
        print(f"[INFO] Categories: {len(self.categories)}, Vocab size: {len(self.vocab)}")
        return self
    
    def predict(self, text):
        """Predict category and confidence."""
        tokens = self.tokenize(text)
        scores = {}
        total_docs = sum(self.class_freq.values()) or 1
        
        for category in self.categories:
            score = 0.0
            prior = self.class_freq[category] / total_docs
            score += math.log(prior if prior > 0 else 1e-12)
            
            category_total = sum(self.word_freq[category].values()) + len(self.vocab) + 1
            for token in tokens:
                word_count = self.word_freq[category].get(token, 0) + 1
                score += math.log(word_count / category_total)
            scores[category] = score
        
        best_category = max(scores, key=scores.get) if scores else None
        if not best_category:
            return None, 0.0
        
        max_score = max(scores.values())
        exp_scores = {cat: math.exp(val - max_score) for cat, val in scores.items()}
        total_exp = sum(exp_scores.values()) or 1.0
        confidence = exp_scores[best_category] / total_exp
        return best_category, confidence
    
    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[INFO] Model saved to {filepath}")


# Sample training data for common job roles
SAMPLE_DATA = [
    # Software Engineer samples
    ("I am a software engineer with 5 years experience in Python and Java. I have built REST APIs and microservices.", "SOFTWARE-ENGINEER"),
    ("Developed web applications using JavaScript, React and Node.js. Strong backend development skills.", "SOFTWARE-ENGINEER"),
    ("C++ programmer with expertise in system design and algorithms. 8 years in software development.", "SOFTWARE-ENGINEER"),
    
    # Data Scientist samples
    ("Data scientist with expertise in machine learning, Python, and statistical analysis. Built prediction models.", "DATA-SCIENTIST"),
    ("Analytics professional skilled in SQL, Tableau, and data visualization. 4 years in BI.", "DATA-SCIENTIST"),
    ("PhD in Data Science. Experience with deep learning, TensorFlow, and big data processing.", "DATA-SCIENTIST"),
    
    # Cloud Engineer samples
    ("Cloud architect experienced in AWS, Docker, and Kubernetes. Built scalable infrastructure.", "CLOUD-ENGINEER"),
    ("DevOps engineer with 6 years in CI/CD, Azure, and infrastructure automation.", "CLOUD-ENGINEER"),
    ("AWS certified specialist in cloud migration and serverless architecture.", "CLOUD-ENGINEER"),
    
    # Data Analyst samples
    ("Business analyst with 5 years in data analysis, SQL queries, and Excel modeling.", "DATA-ANALYST"),
    ("Analytics engineer skilled in data warehousing, ETL pipelines, and reporting.", "DATA-ANALYST"),
    
    # Business Analyst samples
    ("Business analyst with requirements gathering, stakeholder management, and process improvement expertise.", "BUSINESS-ANALYST"),
    ("Consultant with 7 years in business process optimization and strategic planning.", "BUSINESS-ANALYST"),
]

def create_model():
    """Create and save the classifier model."""
    # Prepare data
    X = [text for text, _ in SAMPLE_DATA]
    y = [label for _, label in SAMPLE_DATA]
    
    # Train model
    clf = SimpleNaiveBayesClassifier()
    clf.train(X, y)
    
    # Create models directory
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, "resume_classifier.pkl")
    clf.save(model_path)
    
    print(f"\n[SUCCESS] Model created with {len(clf.categories)} categories")
    print(f"Categories: {sorted(clf.categories)}")
    
    # Test model
    test_text = "I have 5 years as a Python software engineer building APIs and microservices on AWS"
    pred_cat, conf = clf.predict(test_text)
    print(f"\n[TEST] Sample prediction:")
    print(f"  Text: {test_text[:60]}...")
    print(f"  Predicted: {pred_cat} (confidence: {conf:.2%})")


if __name__ == "__main__":
    create_model()
