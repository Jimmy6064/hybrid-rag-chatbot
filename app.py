# app.py
# Hybrid RAG Chatbot (RAG + LLM + live APIs)
# Save as app.py
# Run: streamlit run app.py

import os
import tempfile
import pickle
import time
from pathlib import Path

import streamlit as st
import pdfplumber
import requests
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

# load .env
load_dotenv()

# ---------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_APIKEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", 0.60))
INDEX_PATH = Path("index.faiss")
METAS_PATH = Path("metas.pkl")

# ---------- Optional LLM fallback: OpenAI ----------
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    import openai
    openai.api_key = OPENAI_API_KEY

# ---------- Streamlit page ----------
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("🔎 Hybrid RAG Chatbot — Docs + Live APIs + LLM")

# ---------- Utility: embedding model ----------
@st.cache_resource
def load_embed_model():
    model = SentenceTransformer(EMBED_MODEL)
    return model

model = load_embed_model()
EMBED_DIM = model.get_sentence_embedding_dimension()

# ---------- Utility: chunker ----------
def chunk_text(text, max_words=220, overlap_words=40):
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_words = 0
    for s in sents:
        w = len(s.split())
        if cur_words + w <= max_words:
            cur.append(s); cur_words += w
        else:
            chunks.append(" ".join(cur))
            # start with overlap
            overlap = " ".join(" ".join(cur).split()[-overlap_words:])
            cur = [overlap, s] if overlap else [s]
            cur_words = len(" ".join(cur).split())
    if cur:
        chunks.append(" ".join(cur))
    return [c.strip() for c in chunks if len(c.strip())>20]

# ---------- Ingest PDF ----------
def ingest_pdf(file_bytes, filename):
    # file_bytes: BytesIO or path
    metas = []
    # pdfplumber can accept file path; create temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    if hasattr(file_bytes, "read"):
        tmp.write(file_bytes.read())
    else:
        tmp.write(file_bytes)
    tmp.flush()
    try:
        with pdfplumber.open(tmp.name) as pdf:
            full_text = ""
            for pno, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    full_text += f"\n\n[PAGE {pno}]\n" + page_text
        chunks = chunk_text(full_text)
        for i, c in enumerate(chunks):
            metas.append({"id": f"{filename}__{i}", "source": filename, "text": c})
    finally:
        tmp.close()
    return metas

# ---------- Index management ----------
@st.cache_resource
def load_or_create_index():
    if INDEX_PATH.exists() and METAS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(METAS_PATH, "rb") as f:
            metas = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(EMBED_DIM)
        metas = []
    return index, metas

index, metas = load_or_create_index()

def persist_index():
    faiss.write_index(index, str(INDEX_PATH))
    with open(METAS_PATH, "wb") as f:
        pickle.dump(metas, f)

# ---------- Search / Retrieve ----------
def search_index(query, top_k=10):
    if index.ntotal == 0:
        return []
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({"score": float(dist), "meta": metas[idx]})
    return results

def l2_to_score(dist):
    # smaller L2 dist -> higher similarity-like score
    return 1.0 / (1.0 + dist)

# ---------- Live API helpers ----------
def looks_like_live_query(q):
    ql = q.lower()
    if any(w in ql for w in ["weather", "temperature", "forecast", "rain", "sunny"]):
        return "weather"
    if any(w in ql for w in ["news", "latest", "headlines"]):
        return "news"
    return None

def call_openweather(city="New Delhi"):
    if not OPENWEATHER_KEY:
        return "OpenWeatherMap API key not set. Set OPENWEATHER_APIKEY in .env or streamlit secrets."
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        desc = j["weather"][0]["description"]
        temp = j["main"]["temp"]
        hum = j["main"]["humidity"]
        return f"Weather in {city}: {desc}. Temp: {temp}°C. Humidity: {hum}%."
    except Exception as e:
        return f"Weather API error: {e}"

def call_newsapi(query=None):
    # Placeholder: add NewsAPI/GNews integration if you have keys.
    return "News integration not configured. Add NewsAPI/GNews key to enable live news."

# ---------- LLM helpers ----------
def general_llm_generate(prompt, max_tokens=300, temperature=0.2):
    if not USE_OPENAI:
        return "No OpenAI key configured. Add OPENAI_API_KEY in .env or Streamlit secrets to enable LLM fallback."
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return resp.choices[0].message.content

def llm_generate_with_context(query, top_chunks):
    # Build a strict prompt instructing model to use only sources
    srcs = "\n\n".join([f"[{i+1}] SOURCE: {c['source']}\n{c['text']}" for i,c in enumerate(top_chunks)])
    prompt = (
        "You are an assistant that must answer using ONLY the provided SOURCES.\n"
        "If the answer is not present, respond 'I don't know'.\n\n"
        f"SOURCES:\n{srcs}\n\nQUESTION: {query}\n\nAnswer concisely and cite source numbers like [1]."
    )
    return general_llm_generate(prompt)

# ---------- UI Layout ----------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Index & Upload")
    st.markdown("Upload PDFs (➕) to index them. The chatbot will use uploaded files for document-specific answers.")
    uploaded = st.file_uploader("➕ Upload PDF", type=["pdf"], key="file_uploader")
    if st.button("Index uploaded file"):
        if uploaded is None:
            st.warning("Please select a PDF to upload.")
        else:
            with st.spinner("Ingesting PDF and creating embeddings..."):
                metas_new = ingest_pdf(uploaded, uploaded.name)
                texts = [m["text"] for m in metas_new]
                embs = model.encode(texts, convert_to_numpy=True).astype("float32")
                index.add(embs)
                metas.extend(metas_new)
                persist_index()
                st.success(f"Indexed {len(metas_new)} chunks from {uploaded.name}. Total chunks: {len(metas)}")
    if st.button("Clear index (danger!)"):
        index = faiss.IndexFlatL2(EMBED_DIM)
        metas = []
        if INDEX_PATH.exists(): INDEX_PATH.unlink()
        if METAS_PATH.exists(): METAS_PATH.unlink()
        st.warning("Index cleared. Refresh the page.")
    st.markdown("---")
    st.write("Index status:")
    st.write(f"Vectors stored: {index.ntotal}")
    st.write(f"RAG threshold (similarity-like): {RAG_THRESHOLD}")

with col_right:
    st.subheader("Ask anything — docs + live + general")
    q = st.text_input("Search or ask (e.g., 'What is the refund policy?', 'weather in Mumbai')", key="query")
    if st.button("Ask"):
        if not q:
            st.warning("Type a question or search term.")
        else:
            live = looks_like_live_query(q)
            if live == "weather":
                # rudimentary city extraction
                parts = q.split("in")
                city = parts[-1].strip() if len(parts) > 1 else "New Delhi"
                res = call_openweather(city)
                st.subheader("Live API — Weather")
                st.write(res)
            elif live == "news":
                st.subheader("Live API — News")
                st.write(call_newsapi(q))
            else:
                # RAG retrieval
                results = search_index(q, top_k=10)
                if not results:
                    st.info("No documents indexed. Falling back to general LLM.")
                    out = general_llm_generate(q)
                    st.write(out)
                else:
                    best = results[0]
                    score = l2_to_score(best["score"])
                    st.write(f"Top similarity (approx): {score:.3f}")
                    if score >= RAG_THRESHOLD:
                        top_chunks = [r["meta"] for r in results[:5]]
                        out = llm_generate_with_context(q, top_chunks)
                        st.subheader("Answer (from indexed documents)")
                        st.write(out)
                        st.markdown("**Sources used:**")
                        for i, tc in enumerate(top_chunks, start=1):
                            st.write(f"{i}. {tc['source']} — excerpt: {tc['text'][:280]}...")
                    else:
                        st.info("No strong document match; using general LLM fallback.")
                        out = general_llm_generate(q)
                        st.subheader("Answer (LLM fallback)")
                        st.write(out)

# ---------- Footer / tips ----------
st.markdown("---")
st.markdown("**Tips:** Use the ➕ upload to add PDFs. Add `OPENWEATHER_APIKEY` and `OPENAI_API_KEY` (optional) to `.env` or Streamlit secrets for live answers & LLM fallback.")
