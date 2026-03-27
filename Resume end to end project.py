import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ FUNCTIONS ------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


skills_list = [
    "python", "sql", "pandas", "numpy",
    "machine learning", "data analysis",
    "tableau", "power bi"
]


def extract_skills(text):
    found = []
    for skill in skills_list:
        if skill in text:
            found.append(skill)
    return sorted(list(set(found)))


def calculate_similarity(resume_text, job_text):
    tfidf = TfidfVectorizer()
    matrix = tfidf.fit_transform([resume_text, job_text])
    score = cosine_similarity(matrix[0:1], matrix[1:2])
    return score[0][0]


def get_missing_skills(resume_skills, job_skills):
    return [skill for skill in job_skills if skill not in resume_skills]


skill_category = {
    "python": "foundation",
    "sql": "core",
    "machine learning": "core",
    "data analysis": "core",
    "tableau": "tools",
    "power bi": "tools"
}


def generate_roadmap(missing):
    roadmap = {"foundation": [], "core": [], "tools": [], "advanced": []}
    for skill in missing:
        if skill in skill_category:
            category = skill_category[skill]
            roadmap[category].append(skill)
    return roadmap


# ------------------ UI ------------------

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.markdown("### Get your resume score and personalized roadmap 🚀")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

job_description = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if uploaded_file and job_description:

        # Extract & clean
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_description)

        # Skills
        resume_skills = extract_skills(resume_clean)
        job_skills = extract_skills(job_clean)

        # Similarity
        score = calculate_similarity(resume_clean, job_clean)

        # Missing
        missing = get_missing_skills(resume_skills, job_skills)

        # Roadmap
        roadmap = generate_roadmap(missing)

        # ---------------- OUTPUT ----------------

        st.subheader("📊 Match Score")
        st.progress(score)
        st.write(f"{round(score * 100, 2)} %")

        st.subheader("🧠 Resume Skills")
        st.write(", ".join(resume_skills))

        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing))

        st.subheader("📍 Learning Roadmap")

        for category, skills in roadmap.items():
            if skills:
                st.markdown(f"### {category.capitalize()} Skills")
                for s in skills:
                    st.write(f"✔ {s}")
    else:
        st.warning("Please upload resume and enter job description")