from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

print("Preloading SentenceTransformer embedding model...")
embed_model_name = "paraphrase-MiniLM-L6-v2"
embed_model = SentenceTransformer(embed_model_name)
print(f"✅ {embed_model_name} loaded and cached.")

print("\nPreloading Flan-T5 summarization model...")
summarizer_model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
print(f"✅ {summarizer_model_name} loaded and cached.")

# Optional: run pipeline once to initialize weights
print("\nInitializing summarization pipeline...")
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
_ = summarizer("This is a warm-up sentence.", max_length=20)
print("✅ Models are preloaded and ready for Streamlit!")
