import streamlit as st
import re
import fitz  # PyMuPDF for PDF text extraction

# ✅ Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ✅ Function to calculate ATS score
def calculate_ats_score(resume_text, job_description):
    resume_text = resume_text.lower()
    job_description = job_description.lower()

    jd_keywords = re.findall(r'\b\w+\b', job_description)
    resume_words = re.findall(r'\b\w+\b', resume_text)

    jd_keywords = set(jd_keywords)
    resume_words = set(resume_words)

    matched_keywords = jd_keywords.intersection(resume_words)
    missing_keywords = jd_keywords - resume_words

    score = (len(matched_keywords) / len(jd_keywords)) * 100 if len(jd_keywords) > 0 else 0

    return score, matched_keywords, missing_keywords

# ✅ Streamlit UI
st.set_page_config(page_title="ATS Resume Checker", page_icon="📄", layout="wide")

st.title("📄 ATS Resume Checker")
st.write("Upload your **Resume (PDF)** and paste the **Job Description** to get ATS Score, Keyword Match, and Missing Keywords.")

# Upload Resume
uploaded_resume = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])

# Job Description
job_description = st.text_area("Paste Job Description here")

if uploaded_resume is not None and job_description.strip() != "":
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(uploaded_resume)

        score, matched_keywords, missing_keywords = calculate_ats_score(resume_text, job_description)

        st.subheader("📊 ATS Score Breakdown")
        st.metric("ATS Score", f"{score:.2f}%")

        st.progress(int(score))

        st.subheader("✅ Matched Keywords")
        st.write(", ".join(matched_keywords) if matched_keywords else "No keywords matched.")

        st.subheader("❌ Missing Keywords")
        st.write(", ".join(missing_keywords) if missing_keywords else "No keywords missing.")

else:
    st.info("👆 Upload a Resume and Paste a Job Description to start analysis.")
