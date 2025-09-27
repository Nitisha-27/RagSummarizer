import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
import os
import hashlib

st.set_page_config(page_title="Ultra-FastSummarizer", page_icon="📄", layout="centered")
st.title("RAGSummarizer")

# -----------------------------
# Helper: cache embeddings
# -----------------------------
def get_cache_path(file):
    file_hash = hashlib.md5(file.read()).hexdigest()
    file.seek(0)  # reset pointer
    return f"cache/{file_hash}_embeddings.npy"

# Ensure cache folder exists
os.makedirs("cache", exist_ok=True)

# -----------------------------
# 1. Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

if uploaded_file is not None:
    # Read PDF
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    
    if not text.strip():
        st.error("Could not extract any text from this PDF.")
    else:
        st.success("PDF loaded successfully!")

        # -----------------------------
        # 2. Split into 5-sentence chunks
        # -----------------------------
        sentences = text.split(". ")
        chunk_size = 5
        chunks = [".".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
        st.write(f"Document split into {len(chunks)} chunks.")

        # -----------------------------
        # 3. Load or compute embeddings
        # -----------------------------
        cache_path = get_cache_path(uploaded_file)

        with st.spinner("Computing or loading embeddings..."):
            embed_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")  # super fast
            if os.path.exists(cache_path):
                embeddings = np.load(cache_path)
            else:
                embeddings = []
                batch_size = 32
                for i in range(0, len(chunks), batch_size):
                    batch_emb = embed_model.encode(chunks[i:i+batch_size], convert_to_numpy=True)
                    embeddings.append(batch_emb)
                embeddings = np.vstack(embeddings)
                np.save(cache_path, embeddings)

        # -----------------------------
        # 4. Store in FAISS
        # -----------------------------
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # -----------------------------
        # 5. Query input
        # -----------------------------
        query = st.text_input("Enter your query", "Summarize this document in one paragraph.")
        if st.button("Generate Summary") and query:
            with st.spinner("Retrieving top chunk and generating summary..."):
                # Retrieve top 1 chunk
                q_emb = embed_model.encode([query], convert_to_numpy=True)
                D, I = index.search(q_emb, k=1)
                top_chunk = chunks[I[0][0]]

                # Truncate to 300 words
                top_chunk_words = top_chunk.split()
                if len(top_chunk_words) > 300:
                    top_chunk = " ".join(top_chunk_words[:300])

                # -----------------------------
                # 6. Summarize using flan-t5-small
                # -----------------------------
                summarizer = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",  # lightweight CPU model
                    device=-1
                )
                prompt = f"Summarize the following text in one concise paragraph:\n\n{top_chunk}"

                generated = summarizer(prompt, max_length=120, do_sample=False)
                summary = generated[0]['generated_text']

                st.subheader("📑 Summary")
                st.write(summary)
