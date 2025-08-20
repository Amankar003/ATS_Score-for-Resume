import re
import spacy
import PyPDF2
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 3: Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

# Step 4: Text cleaning function
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Step 5: Example dataset (resume text + category)
data = {
    "resume_text": [
        "Experienced Data Scientist with skills in Python, Machine Learning, Deep Learning",
        "HR professional with expertise in recruitment, payroll, employee management",
        "Software developer skilled in Java, Spring Boot, REST APIs"
    ],
    "category": ["Data Science", "HR", "Software Engineering"]
}

# Step 6: Vectorization + Model training
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["resume_text"])
y = data["category"]

model = MultinomialNB()
model.fit(X, y)

# Step 7: Test with your resume (PDF)
resume_path = "Rounit_Resume.pdf"  # <-- apna resume file ka naam yaha de
resume_text = extract_text_from_pdf(resume_path)
cleaned_resume = clean_text(resume_text)

# Step 8: Prediction
X_test = vectorizer.transform([cleaned_resume])
prediction = model.predict(X_test)
print("Predicted Category:", prediction[0])

# ================================
# Imports
# ================================
import re, math
from pathlib import Path
import fitz  # PyMuPDF
import docx2txt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# Download NLTK stopwords
nltk.download("stopwords")
nltk.download("punkt")

# ================================
# Globals
# ================================
STOP = set(stopwords.words('english'))
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

HEADINGS = [
    "summary","objective","education","experience","work experience","projects",
    "skills","certifications","achievements","publications"
]

HARD_SKILLS = {
    "python","sql","machine learning","deep learning","pandas","numpy","scikit-learn",
    "tensorflow","pytorch","nlp","computer vision","data analysis","statistics",
    "docker","kubernetes","aws","gcp","azure","spark","hadoop","power bi","tableau"
}

SOFT_SKILLS = {"communication","leadership","teamwork","problem solving","critical thinking"}

# ================================
# Resume text extractor
# ================================
def extract_text(path):
    p = Path(path)
    if p.suffix.lower()==".pdf":
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
        return text
    elif p.suffix.lower() in [".docx",".doc"]:
        return docx2txt.process(path)
    else:
        return Path(path).read_text(errors="ignore")

def clean(t):
    return re.sub(r'\s+', ' ', t)

# ================================
# Feature functions
# ================================
def section_presence(text):
    t = text.lower()
    return sum(1 for h in HEADINGS if re.search(rf"\b{re.escape(h)}\b", t))

def bullets_ratio(text):
    lines = text.splitlines()
    if not lines: return 0
    bullets = sum(1 for ln in lines if re.match(r'^\s*[\-\•\*]', ln))
    return bullets/len(lines)

def length_ok(text):
    words = len(text.split())
    return 1.0 if 350<=words<=900 else max(0, min(1, 1 - abs(words-600)/600))

def contact_found(text):
    email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone = re.search(r"\+?\d[\d \-\(\)]{8,}\d", text)
    linkedin = re.search(r"linkedin\.com/in/\S+", text.lower())
    return sum(x is not None for x in [email,phone,linkedin])/3

def skill_hits(text, vocab):
    t = text.lower()
    hits = sum(1 for s in vocab if re.search(rf"\b{re.escape(s)}\b", t))
    return hits, len(vocab)

def jd_relevance(resume_text, jd_text):
    emb_r = MODEL.encode(resume_text, convert_to_tensor=True, normalize_embeddings=True)
    emb_j = MODEL.encode(jd_text, convert_to_tensor=True, normalize_embeddings=True)
    cos = float(util.cos_sim(emb_r, emb_j))
    # TF-IDF overlap (rough)
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = vect.fit_transform([resume_text, jd_text]).toarray()
    tfidf_sim = (X[0] @ X[1]) / (math.sqrt((X[0]**2).sum()) * math.sqrt((X[1]**2).sum()) + 1e-9)
    cos_n = (cos + 1)/2  # normalize [-1,1] to [0,1]
    return 0.7*cos_n + 0.3*tfidf_sim

# ================================
# Main scoring function
# ================================
def ats_score(resume_path, jd_text):
    raw = extract_text(resume_path)
    text = clean(raw)

    # Parsability
    pars = 0.4*contact_found(text) + 0.3*(1 if len(text)>200 else 0) + 0.3*(1 - text.count("  ")/max(1,len(text)))
    pars = max(0, min(1, pars))

    # Structure
    struct = 0.45*(section_presence(text)/6) + 0.35*bullets_ratio(text) + 0.20*length_ok(text)
    struct = max(0, min(1, struct))

    # Keywords
    hs, htot = skill_hits(text, HARD_SKILLS)
    ss, stot = skill_hits(text, SOFT_SKILLS)
    kw = 0.8*(hs/max(1,htot)) + 0.2*(ss/max(1,stot))
    kw = max(0, min(1, kw))

    # Relevance
    rel = jd_relevance(text, jd_text)

    # Weighted final score
    final = 100*(0.25*pars + 0.20*struct + 0.15*kw + 0.40*rel)
    breakdown = {
        "parsability_25": round(100*pars*0.25,2),
        "structure_20": round(100*struct*0.20,2),
        "keywords_15": round(100*kw*0.15,2),
        "relevance_40": round(100*rel*0.40,2),
        "total": round(final,2)
    }
    # missing keywords
    missing = [k for k in HARD_SKILLS if k not in text.lower()]
    return breakdown, missing[:10]

# ================================
# Example usage
# ================================
jd_text = """We are looking for a Machine Learning Engineer with Python, SQL, ML, DL, PyTorch, AWS, deployment experience."""
resume_file = "Rounit_Resume.pdf"   # <-- apna resume ka path do

score, missing = ats_score(resume_file, jd_text)

print("Score Breakdown:", score)
print("Missing Keywords:", missing)
