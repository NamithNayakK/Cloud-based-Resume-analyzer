"""
Resume Classification Model Training Script
Trains a simple text classifier on resume data to categorize resumes by job category.
No heavy dependencies - uses pure Python for compatibility.
"""

import csv
import math
import pickle
import re
from collections import Counter
import os

# Path configuration
CSV_PATH = r"c:\Users\pramukh\Downloads\Resume.csv"
MODEL_OUTPUT_DIR = r"d:\Resume\cloud-resume-analyzer\backend\mlService\models"

# Create model directory if missing
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


class SimpleNaiveBayesClassifier:
    """Lightweight Naive Bayes text classifier for resume categorization."""

    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
        "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them",
        "this", "to", "was", "were", "with", "will", "would", "can", "could", "should", "may",
        "about", "after", "before", "during", "than", "then", "there", "these", "those", "through",
        "over", "under", "up", "down", "out", "off", "also", "more", "most", "many", "much",
    }

    ALLOWED_CATEGORIES = {
        "ACCOUNTANT", "ADVOCATE", "AGRICULTURE", "APPAREL", "ARTS", "AUTOMOBILE", "AVIATION",
        "BANKING", "BPO", "BUSINESS-DEVELOPMENT", "CHEF", "CONSTRUCTION", "CONSULTANT", "DESIGNER",
        "DIGITAL-MEDIA", "ENGINEERING", "FINANCE", "FITNESS", "HEALTHCARE", "HR",
        "INFORMATION-TECHNOLOGY", "PUBLIC-RELATIONS", "SALES", "TEACHER", "SOFTWARE-ENGINEER",
        "CLOUD-ENGINEER",
    }

    CATEGORY_ALIASES = {
        "BUSINESS DEVELOPMENT": "BUSINESS-DEVELOPMENT",
        "DIGITAL MEDIA": "DIGITAL-MEDIA",
        "PUBLIC RELATIONS": "PUBLIC-RELATIONS",
        "INFORMATION TECHNOLOGY": "INFORMATION-TECHNOLOGY",
        "SOFTWARE ENGINEER": "SOFTWARE-ENGINEER",
        "CLOUD ENGINEER": "CLOUD-ENGINEER",
        "HUMAN RESOURCES": "HR",
    }
    
    def __init__(self):
        self.class_freq = {}  # count of each category
        self.word_freq = {}   # word counts per category - using dict instead of defaultdict
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
        """Apply light stemming so similar words share counts."""
        token = token.strip()
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        for suffix in ("ingly", "edly", "ing", "edly", "ed", "ers", "er", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[: -len(suffix)]
        return token
    
    def train(self, X, y):
        """Train the classifier.
        Args:
            X: list of resume texts
            y: list of categories
        """
        print(f"[INFO] Training on {len(X)} resumes...")
        
        for text, category in zip(X, y):
            self.categories.add(category)
            self.class_freq[category] = self.class_freq.get(category, 0) + 1
            
            # Initialize word_freq for category if missing
            if category not in self.word_freq:
                self.word_freq[category] = {}
            
            tokens = self.tokenize(text)
            for token in tokens:
                self.vocab.add(token)
                self.word_freq[category][token] = self.word_freq[category].get(token, 0) + 1
                self.total_words += 1
        
        print(f"[INFO] Training complete!")
        print(f"[INFO] Categories: {sorted(self.categories)}")
        print(f"[INFO] Vocab size: {len(self.vocab)}")
        return self
    
    def predict(self, text):
        """Predict category for resume text."""
        tokens = self.tokenize(text)
        scores = {}
        total_docs = sum(self.class_freq.values()) or 1
        
        for category in self.categories:
            # Use log probabilities to avoid floating-point underflow.
            score = 0.0
            prior = self.class_freq[category] / total_docs
            score += math.log(prior if prior > 0 else 1e-12)

            # Likelihood (add-one smoothing)
            category_total = sum(self.word_freq[category].values()) + len(self.vocab) + 1
            for token in tokens:
                word_count = self.word_freq[category].get(token, 0) + 1
                score += math.log(word_count / category_total)
            scores[category] = score

        best_category = max(scores, key=scores.get)
        raw_scores = list(scores.values())
        max_score = max(raw_scores)
        exp_scores = {cat: math.exp(val - max_score) for cat, val in scores.items()}
        total_exp = sum(exp_scores.values()) or 1.0
        confidence = exp_scores[best_category] / total_exp
        return best_category, confidence
    
    def save(self, filepath):
        """Save model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"[INFO] Model saved to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"[INFO] Model loaded from {filepath}")
        return model


def clean_category(cat):
    """Keep only clean job categories from the dataset."""
    if not cat:
        return None
    cat = str(cat).strip()
    cat = re.sub(r'\s+', ' ', cat)
    if '<' in cat or '>' in cat or 'http' in cat.lower() or '/' in cat or '"' in cat or len(cat) > 35:
        return None
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9&\-\. ]{1,34}', cat):
        return None
    cleaned = cat.upper().replace(' ', '-')
    cleaned = SimpleNaiveBayesClassifier.CATEGORY_ALIASES.get(cleaned.replace('-', ' '), cleaned)
    if cleaned not in SimpleNaiveBayesClassifier.ALLOWED_CATEGORIES:
        return None
    return cleaned


def load_resume_data(csv_path, limit=None):
    """Load resume data from CSV."""
    X, y = [], []
    count = 0
    skipped = 0
    
    print(f"[INFO] Loading resume data from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            resume_text = row.get('Resume_str', '')
            category = row.get('Category', '')
            
            # Clean category
            clean_cat = clean_category(category)
            
            if resume_text and clean_cat:
                X.append(resume_text)
                y.append(clean_cat)
                count += 1
                
                if limit and count >= limit:
                    break
            else:
                skipped += 1
    
    print(f"[INFO] Loaded {count} resumes (skipped {skipped} with missing data)")
    return X, y


def train_model(csv_path, output_dir, limit=None):
    """Main training function."""
    try:
        # Load data
        X, y = load_resume_data(csv_path, limit=limit)
        
        if not X:
            print("[ERROR] No resume data found!")
            return False
        
        # Create and train classifier
        clf = SimpleNaiveBayesClassifier()
        clf.train(X, y)
        
        # Save model
        model_path = os.path.join(output_dir, "resume_classifier.pkl")
        clf.save(model_path)
        
        # Print statistics
        print(f"\n[INFO] === Training Statistics ===")
        print(f"Total resumes: {len(X)}")
        print(f"Categories: {len(clf.categories)}")
        category_dist = Counter(y)
        for cat, count in sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {cat}: {count}")
        print(f"Vocabulary size: {len(clf.vocab)}")
        
        return True
    
    except Exception as exc:
        print(f"[ERROR] Training failed: {exc}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Train on full dataset or subset
    success = train_model(CSV_PATH, MODEL_OUTPUT_DIR, limit=None)
    
    if success:
        print("\n[SUCCESS] Model training completed!")
        
        # Test the model
        clf = SimpleNaiveBayesClassifier.load(os.path.join(MODEL_OUTPUT_DIR, "resume_classifier.pkl"))
        test_text = "Experienced Python developer with 5 years in software engineering and cloud infrastructure"
        pred_category, confidence = clf.predict(test_text)
        print(f"\n[TEST] Sample prediction:")
        print(f"  Text: {test_text[:50]}...")
        print(f"  Category: {pred_category}")
        print(f"  Confidence: {confidence:.6f}")
    else:
        print("\n[ERROR] Model training failed!")
        exit(1)
