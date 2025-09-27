import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="RAG PDF Summarizer", page_icon="📄", layout="centered")
st.title("📄 RAG PDF Summarizer with Instruction-Tuned Model")

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
        # 2. Split into chunks
        # -----------------------------
        # Split by sentences or approximate ~300 words per chunk
        chunks = text.split(". ")
        st.write(f"Document split into {len(chunks)} chunks.")

        # -----------------------------
        # 3. Compute embeddings
        # -----------------------------
        with st.spinner("Computing embeddings..."):
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embed_model.encode(chunks, convert_to_numpy=True)

        # -----------------------------
        # 4. Store in FAISS
        # -----------------------------
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # -----------------------------
        # 5. Ask query
        # -----------------------------
        query = st.text_input("Enter your query", "Summarize this document in one paragraph.")
        if st.button("Generate Summary") and query:
            with st.spinner("Retrieving relevant chunks and generating summary..."):
                # Retrieve top 3 relevant chunks
                q_emb = embed_model.encode([query], convert_to_numpy=True)
                D, I = index.search(q_emb, k=3)
                top_chunks = [chunks[i] for i in I[0]]

                # Combine top chunks as context
                context = " ".join(top_chunks)

                # -----------------------------
                # 6. Generate summary with instruction-tuned model
                # -----------------------------
                summarizer = pipeline("text2text-generation", model="google/flan-t5-large")
                prompt = f"Summarize the following text in one concise paragraph:\n\n{context}"

                generated = summarizer(prompt, max_length=200, do_sample=False)
                summary = generated[0]['generated_text']

                st.subheader("📑 Summary")
                st.write(summary)
