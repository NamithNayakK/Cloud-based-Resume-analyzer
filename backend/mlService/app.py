import re
import os
import math
import pickle
from collections import Counter
from flask import Flask, request, jsonify

# Optional embeddings support
EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDINGS_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDINGS_AVAILABLE = True
    print('[INFO] SentenceTransformer loaded for semantic matching.')
except Exception as exc:
    EMBEDDINGS_AVAILABLE = False
    print(f'[WARN] SentenceTransformer not available: {exc}. Falling back to keyword matching.')

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

ROLE_RECOMMENDATIONS = {
    "software": ["Software Engineer", "Backend Developer", "API Developer"],
    "data": ["Data Analyst", "Business Intelligence Analyst", "Data Scientist"],
    "cloud": ["Cloud Engineer", "DevOps Engineer", "Platform Engineer"],
    "web": ["Frontend Developer", "Full Stack Developer", "UI Engineer"],
    "mobile": ["Mobile App Developer", "Android Developer", "iOS Developer"],
    "finance": ["Financial Analyst", "Accounting Associate", "Operations Analyst"],
    "support": ["Customer Support Specialist", "Operations Associate", "Client Success Associate"],
}

ROLE_SKILL_GAPS = {
    "software": ["python", "api", "rest", "docker", "backend", "microservices", "cloud", "testing"],
    "data": ["sql", "excel", "power bi", "tableau", "statistics", "dashboarding", "etl", "analytics"],
    "cloud": ["aws", "azure", "docker", "kubernetes", "ci/cd", "terraform", "observability", "linux"],
    "web": ["javascript", "react", "html", "css", "accessibility", "performance", "api integration"],
    "mobile": ["android", "ios", "flutter", "kotlin", "swift", "mobile testing"],
    "finance": ["financial modeling", "excel", "forecasting", "reporting", "accounting", "risk analysis"],
    "support": ["customer communication", "ticketing", "sla", "troubleshooting", "crm"],
}

# Precompute role-family embeddings if embeddings are available
ROLE_FAMILY_EMBEDDINGS = {}
if EMBEDDINGS_AVAILABLE:
    try:
        for family, roles in ROLE_RECOMMENDATIONS.items():
            # Build a representative description combining role names and expected skills
            skills = ' '.join(ROLE_SKILL_GAPS.get(family, []))
            text = f"{family} {' '.join(roles)} {skills}"
            ROLE_FAMILY_EMBEDDINGS[family] = EMBEDDINGS_MODEL.encode(text)
    except Exception as exc:
        print(f'[WARN] Failed to precompute role embeddings: {exc}')


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


def cosine_similarity(a, b):
    try:
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception:
        return 0.0


def compute_semantic_role_scores(resume_text: str):
    """Return semantic similarity scores for each role family using embeddings.
    Returns list of dicts with role family, score (0-1), and evidence snippet.
    """
    results = []
    if not EMBEDDINGS_AVAILABLE:
        return results

    try:
        emb = EMBEDDINGS_MODEL.encode(resume_text)
        for family, fam_emb in ROLE_FAMILY_EMBEDDINGS.items():
            sim = cosine_similarity(emb, fam_emb)
            # find short evidence: top sentence from resume matching any family keyword
            sentences = [s.strip() for s in re.split(r'[\.\n]', resume_text) if s.strip()]
            evidence = ''
            for sent in sentences[:8]:
                if any(k in sent.lower() for k in JOB_FAMILY_KEYWORDS.get(family, [])):
                    evidence = sent
                    break
            results.append({
                'family': family,
                'score': round(float(sim), 4),
                'evidence': evidence,
            })
        # sort descending
        results.sort(key=lambda r: r['score'], reverse=True)
    except Exception as exc:
        print(f'[WARN] Semantic scoring failed: {exc}')

    return results


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


def infer_job_family_from_resume(resume_text: str, extracted: dict):
    """Infer the closest job family using the resume text and extracted skills."""
    lowered = f"{resume_text} {' '.join(extracted.get('skills', []))}".lower()
    for family, keywords in JOB_FAMILY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return family
    return None


def normalize_role_name(category: str):
    if not category:
        return ""
    return category.replace("-", " ").replace("_", " ").title()


def role_family_from_category(category: str):
    if not category:
        return None
    normalized = category.upper().replace(" ", "-")
    if any(token in normalized for token in ["DATA", "ANALYT", "SCIENCE"]):
        return "data"
    if any(token in normalized for token in ["CLOUD", "DEVOPS", "INFRA"]):
        return "cloud"
    if any(token in normalized for token in ["WEB", "FRONTEND", "BACKEND", "FULLSTACK"]):
        return "web"
    if any(token in normalized for token in ["MOBILE", "ANDROID", "IOS", "FLUTTER"]):
        return "mobile"
    if any(token in normalized for token in ["FINANCE", "ACCOUNT", "BANK"]):
        return "finance"
    if any(token in normalized for token in ["SUPPORT", "BPO", "SERVICE", "HELPDESK"]):
        return "support"
    return "software"


def build_resume_quality_score(resume_text: str, extracted: dict, predicted_category: str, category_confidence: float):
    """Score the resume on quality signals when no JD is supplied."""
    score = 20
    lowered = resume_text.lower()

    if extracted.get("name"):
        score += 8
    if extracted.get("email"):
        score += 8
    if extracted.get("phone"):
        score += 8
    if extracted.get("education"):
        score += 8
    if extracted.get("skills"):
        score += min(18, len(extracted.get("skills", [])) * 3)
    if extracted.get("totalExperienceYears", 0) > 0:
        score += min(15, extracted.get("totalExperienceYears", 0) * 2)

    if re.search(r'\b(project|built|implemented|developed|designed|led|improved|reduced|increased|delivered)\b', lowered):
        score += 10
    if re.search(r'\b\d+%|\b\d+\s*(?:projects?|users?|clients?|customers?|revenue|cost|sales)\b', lowered):
        score += 12
    if re.search(r'\b(summary|profile|objective)\b', lowered):
        score += 4

    if predicted_category:
        score += min(10, int(round(category_confidence * 10)))

    if len(resume_text.split()) < 120:
        score -= 6

    return max(10, min(100, score))


def build_role_recommendations(resume_text: str, extracted: dict, predicted_category: str, category_confidence: float):
    family = infer_job_family_from_resume(resume_text, extracted) or role_family_from_category(predicted_category)
    choices = ROLE_RECOMMENDATIONS.get(family, ROLE_RECOMMENDATIONS["software"])
    skills = extracted.get("skills", [])[:6]

    primary_role = normalize_role_name(predicted_category) if predicted_category else choices[0]
    if not primary_role:
        primary_role = choices[0]

    role_recs = [
        {
            "role": primary_role,
            "confidence": round(max(0.55, category_confidence), 2),
            "reason": [
                f"Resume signals align with {family or 'software'} work.",
                f"Detected skills: {', '.join(skills) if skills else 'general technical background'}.",
            ],
            "bestFor": [family or "software"],
        }
    ]

    for role in choices[1:3]:
        role_recs.append({
            "role": role,
            "confidence": 0.58,
            "reason": [f"Shares a similar skill base with the {family or 'software'} family."],
            "bestFor": [family or "software"],
        })

    return role_recs


def build_resume_issues(resume_text: str, extracted: dict):
    issues = []
    lowered = resume_text.lower()

    if not extracted.get("name"):
        issues.append("Your resume does not clearly expose a candidate name in the extracted text.")
    if not extracted.get("email") or not extracted.get("phone"):
        issues.append("Contact details are incomplete or not easy to detect.")
    if len(extracted.get("skills", [])) < 3:
        issues.append("Only a small number of concrete skills were detected.")
    if not extracted.get("education"):
        issues.append("No education section was clearly detected.")
    if not re.search(r'\b(project|built|implemented|developed|designed|led|improved|reduced|increased|delivered)\b', lowered):
        issues.append("No strong action verbs or project outcomes were detected.")
    if not re.search(r'\b\d+%|\b\d+\s*(?:projects?|users?|clients?|customers?|revenue|cost|sales)\b', lowered):
        issues.append("The resume does not show enough measurable impact.")
    if len(resume_text.split()) < 120:
        issues.append("The resume text looks short; important achievements may be missing.")

    return issues[:6]


def build_skill_gaps(resume_text: str, extracted: dict, family: str):
    gaps = []
    lowered = resume_text.lower()
    keywords = ROLE_SKILL_GAPS.get(family or "software", ROLE_SKILL_GAPS["software"])
    skills = {skill.lower() for skill in extracted.get("skills", [])}

    for keyword in keywords:
        if keyword not in lowered and keyword not in skills:
            gaps.append(keyword)

    if not gaps:
        gaps.extend(["quantified achievements", "project outcomes", "strong technical summary"])

    return gaps[:8]


def build_strengths(extracted: dict, predicted_category: str):
    strengths = []
    skills = extracted.get("skills", [])
    if skills:
        strengths.append(f"Detected skills: {', '.join(skills[:6])}.")
    if extracted.get("education"):
        strengths.append(f"Education signals found: {', '.join(extracted.get('education', []))}.")
    if extracted.get("totalExperienceYears", 0):
        strengths.append(f"Experience signal: about {extracted.get('totalExperienceYears', 0)} years.")
    if predicted_category:
        strengths.append(f"Best-fit role signal: {normalize_role_name(predicted_category)}.")
    return strengths[:5]


def build_improvement_plan(issues: list, extracted: dict):
    plan = []
    if any("Contact details" in issue for issue in issues):
        plan.append("Add your name, phone number, and email at the top in a clean header.")
    if any("measurable impact" in issue for issue in issues):
        plan.append("Rewrite project bullets with numbers, percentages, or business outcomes.")
    if any("action verbs" in issue for issue in issues):
        plan.append("Use strong verbs such as built, automated, reduced, shipped, and improved.")
    if any("short" in issue for issue in issues):
        plan.append("Expand the resume with projects, internships, or case studies." )
    if len(extracted.get("skills", [])) < 3:
        plan.append("Add a focused skills section with tools, frameworks, and domain keywords.")
    if not extracted.get("education"):
        plan.append("Include your education section or relevant certifications if available.")
    if not plan:
        plan.append("Polish the summary and keep only role-relevant achievements.")

    return plan[:6]


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
        
        # Extract entities
        extracted = extract_entities(resume_text)
        
        # Predict category using trained model
        predicted_category, category_confidence = predict_category(resume_text)
        if predicted_category:
            extracted["predictedCategory"] = predicted_category
            extracted["categoryConfidence"] = round(category_confidence, 4)

        job_family = extract_job_family(job_description) if job_description.strip() else infer_job_family_from_resume(resume_text, extracted)
        if job_family:
            extracted["jobFamily"] = job_family

        if job_description.strip():
            # Compute similarity when a JD is supplied.
            score, matched, missing = compute_tfidf_similarity(resume_text, job_description)
            print(f"[DEBUG] Similarity score: {score}%, matched: {len(matched)}, missing: {len(missing)}")
            score = min(100, score + score_category_alignment(predicted_category, job_family))
            if score < 10 and resume_text.strip():
                score = 10
        else:
            matched = extracted.get("skills", [])[:8]
            missing = build_skill_gaps(resume_text, extracted, job_family)
            score = build_resume_quality_score(resume_text, extracted, predicted_category, category_confidence)
        
        # Generate recommendations
        if EMBEDDINGS_AVAILABLE:
            semantic_scores = compute_semantic_role_scores(resume_text)
            role_recommendations = []
            # build role entries from semantic family scores
            for score_entry in semantic_scores[:4]:
                family = score_entry.get('family')
                fam_score = score_entry.get('score', 0.0)
                choices = ROLE_RECOMMENDATIONS.get(family, ROLE_RECOMMENDATIONS['software'])
                # primary role is first choice
                primary = choices[0] if choices else normalize_role_name(predicted_category) or 'Suggested role'
                role_recommendations.append({
                    'role': primary,
                    'confidence': round(float(max(0.35, fam_score)), 2),
                    'reason': [f"Semantic similarity to {family} roles (score {round(fam_score,3)})", f"Evidence: {score_entry.get('evidence') or 'N/A'}"],
                    'bestFor': [family]
                })
                # also include one alternate
                for alt in choices[1:2]:
                    role_recommendations.append({
                        'role': alt,
                        'confidence': round(float(max(0.32, fam_score * 0.9)), 2),
                        'reason': [f"Related to {family} skill base."],
                        'bestFor': [family]
                    })
        else:
            role_recommendations = build_role_recommendations(resume_text, extracted, predicted_category, category_confidence)
        resume_issues = build_resume_issues(resume_text, extracted)
        strengths = build_strengths(extracted, predicted_category)
        improvement_plan = build_improvement_plan(resume_issues, extracted)
        if job_description.strip():
            recommendations = generate_recommendations(score, missing, resume_text)
            if predicted_category and predicted_category not in (job_family or "").upper():
                recommendations.append(f"Your resume looks closer to {predicted_category}. Consider tailoring it for the target role.")
        else:
            recommendations = improvement_plan
        
        return jsonify({
            "score": score,
            "analysisMode": "resume-to-role" if not job_description.strip() else "resume-vs-job-description",
            "matchedKeywords": matched[:30],
            "missingKeywords": missing[:30],
            "skillGaps": missing[:30],
            "recommendations": recommendations,
            "roleRecommendations": role_recommendations,
            "resumeIssues": resume_issues,
            "strengthSignals": strengths,
            "improvementPlan": improvement_plan,
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
