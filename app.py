import streamlit as st
import joblib
import os

BASE_DIR = os.path.dirname(__file__)

nlp_model = joblib.load(os.path.join(BASE_DIR, "models", "nlp_resume_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))


THRESHOLD = 0.4

def decision_label(p, threshold=0.4):
    if p >= threshold:
        return "✅ Recommend Interview"
    elif p >= threshold - 0.1:
        return "⚠️ Borderline"
    else:
        return "❌ Unlikely"

st.title("📄 AI Resume Screening Demo")
st.write("This demo predicts interview recommendation based on resume text.")

resume_text = st.text_area(
    "Paste Resume Text Here:",
    height=200,
    placeholder="Skills, experience, certifications..."
)

if st.button("Evaluate Resume"):
    if resume_text.strip() == "":
        st.warning("Please enter resume text.")
    else:
        X = tfidf.transform([resume_text])
        prob = nlp_model.predict_proba(X)[0, 1]
        decision = decision_label(prob, THRESHOLD)

        st.subheader("🔍 Prediction Result")
        st.metric("Hire Probability", f"{prob:.2f}")
        st.write("Decision:", decision)

        st.caption(f"Threshold = {THRESHOLD}")
