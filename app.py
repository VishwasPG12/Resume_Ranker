import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text
def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    return cosine_similarity(vectors[0], vectors[1])[0][0] * 100
st.title("Resume Ranker")
resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Paste Job Description")
if st.button("Analyze"):
    if resume and jd:
        resume_text = extract_text_from_pdf(resume)
        score = calculate_similarity(resume_text, jd)
        st.write(f"Based on your Resume and Job description,your Match Score: {score:.2f}%")
        if score >= 75:
            st.success(f"Excellent! Your resume ranks {score:.2f}% for this job.")
        elif score >= 50:
            st.info(f"Good! Your resume ranks {score:.2f}%. Some improvements recommended.")
        else:
            st.warning(f"Your resume ranks only {score:.2f}%. Consider adding more relevant skills.")
    else:
        st.warning("Please upload a resume and paste the job description.")
