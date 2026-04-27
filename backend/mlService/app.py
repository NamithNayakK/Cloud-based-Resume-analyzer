import re
import os
import math
import pickle
from collections import Counter
from flask import Flask, request, jsonify

app = Flask(__name__)


class SimpleNaiveBayesClassifier:
    """Lightweight Naive Bayes text classifier for resume categorization."""
    
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
        return [w for w in text.split() if len(w) > 2]
    
    def predict(self, text):
        """Predict category for resume text."""
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
        
        best_category = max(scores, key=scores.get)
        max_score = max(scores.values())
        exp_scores = {cat: math.exp(val - max_score) for cat, val in scores.items()}
        total_exp = sum(exp_scores.values()) or 1.0
        confidence = exp_scores[best_category] / total_exp
        return best_category, confidence


# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "resume_classifier.pkl")

try:
    with open(MODEL_PATH, 'rb') as f:
        TRAINED_CLASSIFIER = pickle.load(f)
    print(f"[INFO] Trained classifier loaded successfully from {MODEL_PATH}")
    print(f"[INFO] Categories: {len(TRAINED_CLASSIFIER.categories)}, Vocab: {len(TRAINED_CLASSIFIER.vocab)}")
except Exception as exc:
    print(f"[WARN] Could not load trained model: {exc}. Using fallback analysis.")
    TRAINED_CLASSIFIER = None

SKILL_KEYWORDS = {
    "python", "java", "javascript", "node", "react", "sql", "aws", "azure", "docker",
    "kubernetes", "fastapi", "flask", "django", "git", "linux", "html", "css", "mongodb",
    "postgresql", "mysql", "rest", "api", "microservices", "kafka", "redis", "machine learning",
    "data science", "excel", "power bi", "tableau", "cloud", "devops", "spring", "c", "c++"
}

EDUCATION_KEYWORDS = {
    "b.tech", "btech", "m.tech", "mtech", "bachelor", "master", "phd", "degree", "university", "college"
}

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "have",
    "in", "into", "is", "it", "its", "of", "on", "or", "that", "the", "their", "them",
    "this", "to", "was", "were", "with", "will", "would", "can", "could", "should", "may",
    "about", "after", "before", "during", "than", "then", "there", "these", "those", "through",
    "over", "under", "up", "down", "out", "off", "also", "more", "most", "many", "much",
    "experience", "experienced", "skills", "skill", "work", "worked", "working", "responsible",
    "resume", "curriculum", "vitae", "professional", "projects", "project", "years", "year"
}

JOB_FAMILY_KEYWORDS = {
    "software": {"software", "developer", "engineer", "programmer", "coding", "development", "application"},
    "data": {"data", "analytics", "analyst", "analysis", "bi", "etl", "warehouse", "science", "learning"},
    "cloud": {"cloud", "aws", "azure", "gcp", "devops", "docker", "kubernetes", "infra", "infrastructure"},
    "web": {"web", "frontend", "backend", "fullstack", "html", "css", "javascript", "react", "node"},
    "mobile": {"android", "ios", "mobile", "flutter", "react native", "kotlin", "swift"},
    "finance": {"finance", "accounting", "accountant", "bank", "banking", "audit"},
    "support": {"support", "helpdesk", "customer", "service", "call center", "bpo"},
}


def normalize_token(token: str):
    """Apply light stemming so similar words share counts."""
    token = token.strip().lower()
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    for suffix in ("ingly", "edly", "ing", "edly", "ed", "ers", "er", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def extract_email(text: str):
    """Extract email from text using regex."""
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return match.group(0) if match else ""


def extract_phone(text: str):
    """Extract phone number from text."""
    match = re.search(r"(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){10,13}", text)
    return re.sub(r"\s+", "", match.group(0)) if match else ""


def extract_years_experience(text: str):
    """Extract years of experience from text."""
    years = re.findall(r"(\d+)\+?\s*(?:years|yrs)", text.lower())
    if not years:
        return 0
    return max(int(y) for y in years)


def tokenize(text: str):
    """Convert text to lowercase tokens, removing punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s+]', ' ', text)
    tokens = []
    for word in text.split():
        word = normalize_token(word)
        if len(word) > 2 and word not in STOP_WORDS:
            tokens.append(word)
    return tokens


def compute_tfidf_similarity(resume_text: str, job_description: str):
    """
    Compute TF-IDF based similarity score using basic token counting.
    Returns score 0-100 based on keyword overlap.
    """
    resume_tokens = tokenize(resume_text)
    jd_tokens = tokenize(job_description)
    
    if not jd_tokens:
        return 0, [], []
    
    resume_counter = Counter(resume_tokens)
    jd_counter = Counter(jd_tokens)
    jd_text = job_description.lower()
    resume_text_lower = resume_text.lower()
    
    # Find matched and missing keywords
    matched = []
    missing = []
    
    for token, count in jd_counter.items():
        if token in resume_counter or token in resume_text_lower:
            matched.append(token)
        else:
            missing.append(token)
    
    # Calculate similarity score
    if len(jd_counter) == 0:
        score = 0
    else:
        overlap_score = len(matched) / len(jd_counter)
        char_overlap = len(set(jd_text.split()) & set(resume_text_lower.split())) / max(len(set(jd_text.split())), 1)
        score = int(round((overlap_score * 0.7 + char_overlap * 0.3) * 100))
        if score == 0 and resume_text.strip():
            score = 5
    
    return score, matched, missing


def extract_entities(text: str):
    """Extract name, email, phone, skills, education, experience years."""
    result = {
        "name": "",
        "email": extract_email(text),
        "phone": extract_phone(text),
        "skills": [],
        "education": [],
        "experience": [],
        "totalExperienceYears": extract_years_experience(text)
    }
    
    lowered = text.lower()
    result["skills"] = sorted([skill for skill in SKILL_KEYWORDS if skill in lowered])
    result["education"] = sorted([edu for edu in EDUCATION_KEYWORDS if edu in lowered])
    
    # Try to extract name from first line or "Name:" pattern
    lines = text.split('\n')
    for line in lines[:5]:
        if 'name' in line.lower() and ':' in line:
            name_part = line.split(':')[1].strip()
            if name_part and len(name_part) < 100:
                result["name"] = name_part
                break
    
    return result


def extract_job_family(job_description: str):
    """Infer a broad job family from the job description."""
    jd = job_description.lower()
    for family, keywords in JOB_FAMILY_KEYWORDS.items():
        if any(keyword in jd for keyword in keywords):
            return family
    return None


def generate_recommendations(score: int, missing_keywords: list, resume_text: str):
    """Generate recommendations based on score and gaps."""
    recommendations = []
    
    if score < 40:
        recommendations.append("Resume has low similarity with the job description. Add role-specific content.")
    elif score < 60:
        recommendations.append("Good foundation. Strengthen alignment by adding more job-specific keywords.")
    else:
        recommendations.append("Strong keyword alignment. Focus on improving quantified achievements.")
    
    if missing_keywords:
        top_missing = missing_keywords[:8]
        recommendations.append(f"Add missing keywords: {', '.join(top_missing)}")
    else:
        recommendations.append("Add more technical keywords and project outcomes to improve the match.")
    
    if not re.search(r'project|achievement|accomplished|results?|built|implemented|developed', resume_text, re.IGNORECASE):
        recommendations.append("Add quantified achievements and project outcomes to strengthen impact.")
    
    return recommendations


def predict_category(resume_text: str):
    """Predict job category using trained model."""
    if TRAINED_CLASSIFIER is None:
        return None, 0.0
    
    try:
        category, confidence = TRAINED_CLASSIFIER.predict(resume_text)
        # Filter out obviously bad predictions (messy categories from CSV)
        if len(category) > 50 or '<' in category or '&' in category:
            return None, 0.0
        return category, confidence
    except Exception as exc:
        print(f"[WARN] Prediction failed: {exc}")
        return None, 0.0


def score_category_alignment(predicted_category: str, job_family: str):
    """Convert predicted category to a match boost for the job family."""
    if not predicted_category or not job_family:
        return 0

    cat = predicted_category.upper().replace(' ', '-').replace('_', '-')
    if job_family == "software" and any(term in cat for term in ["ENGINEER", "DEVELOPER", "SOFTWARE", "IT", "TECH"]):
        return 20
    if job_family == "data" and any(term in cat for term in ["DATA", "ANALYST", "SCIENCE", "ANALYTICS"]):
        return 20
    if job_family == "cloud" and any(term in cat for term in ["DEVOPS", "CLOUD", "ENGINEER", "INFRA"]):
        return 20
    if job_family == "web" and any(term in cat for term in ["WEB", "FRONTEND", "BACKEND", "FULLSTACK", "SOFTWARE"]):
        return 18
    if job_family == "finance" and any(term in cat for term in ["FINANCE", "ACCOUNT", "BANK"]):
        return 18
    return 8 if predicted_category else 0


@app.route("/", methods=["GET"])
def index():
    """Root endpoint to help users discover available API routes."""
    return jsonify({
        "service": "ml-analyzer",
        "status": "ok",
        "message": "Use /health for health checks and /analyze for resume analysis.",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze"
        }
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze resume against job description."""
    try:
        payload = request.get_json(force=True) or {}
        resume_text = payload.get("resumeText", "")
        job_description = payload.get("jobDescription", "")
        
        print(f"[DEBUG] Resume text length: {len(resume_text)} chars")
        print(f"[DEBUG] Job description: '{job_description[:100]}'")
        
        if not resume_text:
            return jsonify({"error": "resumeText is required"}), 400
        
        if len(resume_text.strip()) < 10:
            print(f"[WARN] Resume text too short. Content: {resume_text}")
        
        # Compute similarity
        score, matched, missing = compute_tfidf_similarity(resume_text, job_description)
        print(f"[DEBUG] Similarity score: {score}%, matched: {len(matched)}, missing: {len(missing)}")
        
        # Extract entities
        extracted = extract_entities(resume_text)
        
        # Predict category using trained model
        predicted_category, category_confidence = predict_category(resume_text)
        if predicted_category:
            extracted["predictedCategory"] = predicted_category
            extracted["categoryConfidence"] = round(category_confidence, 4)

        job_family = extract_job_family(job_description)
        if job_family:
            extracted["jobFamily"] = job_family

        # Boost the score with model-based alignment so short JDs still get useful results.
        score = min(100, score + score_category_alignment(predicted_category, job_family))
        if score < 10 and resume_text.strip():
            score = 10
        
        # Generate recommendations
        recommendations = generate_recommendations(score, missing, resume_text)
        if predicted_category and predicted_category not in (job_family or "").upper():
            recommendations.append(f"Your resume looks closer to {predicted_category}. Consider tailoring it for the target role.")
        
        return jsonify({
            "score": score,
            "matchedKeywords": matched[:30],
            "missingKeywords": missing[:30],
            "recommendations": recommendations,
            "extracted": extracted
        })
    except Exception as exc:
        print(f"[ERROR] Analyze failed: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    model_status = "loaded" if TRAINED_CLASSIFIER else "fallback"
    return jsonify({
        "status": "ok",
        "service": "ml-analyzer",
        "model": model_status
    })


if __name__ == "__main__":
    print("[INFO] Starting ML Analyzer service on port 5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)
