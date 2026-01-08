import streamlit as st
from transformers import pipeline
import pdfplumber

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Study Notes Generator",
    page_icon="📖",
    layout="centered"
)

st.title("📖 AI-Powered Study Notes Generator")
st.caption("Summarize text or generate exam-ready notes from PDFs")

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Options")

summary_type = st.sidebar.selectbox(
    "📏 Summary Length",
    ["Short", "Medium", "Detailed"]
)

length_map = {
    "Short": (60, 20),
    "Medium": (120, 40),
    "Detailed": (200, 70)
}

MAX_LEN, MIN_LEN = length_map[summary_type]

# ================= LOAD MODEL =================
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# ================= FUNCTIONS =================
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def summarize_text(text):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=MAX_LEN,
            min_length=MIN_LEN,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)

def to_bullets(summary):
    sentences = summary.split(". ")
    return ["• " + s.strip() for s in sentences if len(s.strip()) > 5]

# ================= USER INPUT =================
user_input = st.text_area("✍️ Enter text to summarize:")

uploaded_file = st.file_uploader(
    "📂 Upload study material (PDF)",
    type=["pdf"]
)

text = ""

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    if text:
        st.subheader("📑 Extracted Text Preview")
        st.text_area("Preview", text[:1200], height=250)
        st.caption(f"📊 Word count: {len(text.split())}")
    else:
        st.error("⚠️ No text found in the PDF.")

# ================= BUTTONS =================
col1, col2 = st.columns(2)

# -------- TEXT SUMMARIZER --------
with col1:
    if st.button("📝 Summarize Text"):
        if not user_input.strip():
            st.warning("Please enter text to summarize.")
        else:
            with st.spinner("Summarizing text..."):
                try:
                    summary = summarize_text(user_input)
                    st.subheader("📝 Text Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# -------- STUDY NOTES GENERATOR --------
with col2:
    if st.button("📌 Generate Study Notes"):
        if not user_input.strip() and not text:
            st.warning("Please enter text or upload a PDF.")
        else:
            with st.spinner("Generating study notes..."):
                try:
                    source_text = text if text else user_input
                    summary = summarize_text(source_text)

                    st.subheader("📚 AI-Generated Study Notes")
                    bullets = to_bullets(summary)

                    for bullet in bullets:
                        st.write(bullet)

                    st.download_button(
                        "⬇️ Download Notes",
                        summary,
                        file_name="AI_Study_Notes.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

# ================= FOOTER =================
st.markdown("---")
st.caption("✨ Built with Streamlit & Hugging Face Transformers")
