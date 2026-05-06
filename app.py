import streamlit as st
import pickle
from utils import clean_text
import time

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI Spam Detector", layout="wide")

# 🌌 PARTICLE BACKGROUND + UI
st.markdown("""
<style>

/* Full background */
body {
    margin: 0;
    overflow: hidden;
}

/* Canvas for particles */
#particles-js {
    position: fixed;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: #0f2027;
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}

/* Glass card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
}

/* Button */
.stButton>button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    background: linear-gradient(90deg, #ff6a00, #ee0979);
    color: white;
    font-size: 18px;
    border: none;
}

/* Circular progress */
.circle {
    width: 150px;
    height: 150px;
    border-radius: 50%;
    background: conic-gradient(#ff4b5c var(--value), #222 var(--value));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    color: white;
    margin: auto;
}

/* Explanation box */
.explain {
    margin-top: 15px;
    padding: 10px;
    background: rgba(255,255,255,0.08);
    border-radius: 10px;
}

</style>

<!-- Particle container -->
<div id="particles-js"></div>

<!-- Particle JS -->
<script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
<script>
particlesJS("particles-js", {
  "particles": {
    "number": { "value": 80 },
    "color": { "value": "#ffffff" },
    "shape": { "type": "circle" },
    "opacity": { "value": 0.5 },
    "size": { "value": 3 },
    "move": { "enable": true, "speed": 2 }
  }
});
</script>
""", unsafe_allow_html=True)

# TITLE
st.markdown("<div class='title'>🚀 AI Spam Detection Dashboard</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

# INPUT
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_input = st.text_area("✉️ Enter Email", height=200)

    if st.button("🔍 Analyze Email"):
        if user_input.strip() == "":
            st.warning("Enter text")
        else:
            with st.spinner("Analyzing..."):
                time.sleep(1)

            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned]).toarray()

            prediction = model.predict(vectorized)[0]
            prob = model.predict_proba(vectorized)[0]
            confidence = int(max(prob) * 100)

            st.session_state.result = (prediction, confidence, user_input)

    st.markdown("</div>", unsafe_allow_html=True)

# RESULT PANEL
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.subheader("📊 Result")

    if "result" in st.session_state:
        prediction, confidence, text = st.session_state.result

        # 🎯 Circular progress
        st.markdown(
            f"<div class='circle' style='--value:{confidence}%'> {confidence}% </div>",
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ SAFE EMAIL")

        # 🧠 AI Explanation
        explanation = ""

        if prediction == 1:
            explanation = "This message looks like spam because it contains promotional or urgent keywords like 'win', 'free', or 'click'."
        else:
            explanation = "This message appears normal and does not contain suspicious patterns."

        st.markdown(f"<div class='explain'>🧠 {explanation}</div>", unsafe_allow_html=True)

    else:
        st.info("Run analysis to see result")

    st.markdown("</div>", unsafe_allow_html=True)