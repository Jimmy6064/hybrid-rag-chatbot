# app.py - Hybrid RAG Chatbot (Streamlit Cloud Safe Version)

import os
import tempfile
import pickle
from pathlib import Path
import streamlit as st
import pdfplumber
import requests
import numpy as np
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from openai import OpenAI

# ----------- Load env ----------- 
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_APIKEY", "")
RAG_THRESHOLD = float(os.getenv("RAG_THRESHOLD", 0.60))

INDEX_PATH = Path("index_embeddings.npy")
METAS_PATH = Path("metas.pkl")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ----------- Embedding Helper (OpenAI) -----------

def embed_texts(texts):
    if not client:
        return None
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embs = [d.embedding for d in resp.data]
    return np.array(embs, dtype="float32")

def embed_query(q):
    e = embed_texts([q])
    return e[0] if e is not None else None

# ----------- Chunker -----------

def chunk_text(text, max_words=200, overlap_words=40):
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_words = 0

    for s in sents:
        w = len(s.split())
        if cur_words + w <= max_words:
            cur.append(s)
            cur_words += w
        else:
            chunks.append(" ".join(cur))
            overlap = " ".join(" ".join(cur).split()[-overlap_words:])
            cur = [overlap, s] if overlap else [s]
            cur_words = len(" ".join(cur).split())

    if cur:
        chunks.append(" ".join(cur))

    return [c for c in chunks if len(c) > 20]

# ----------- PDF Ingest -----------

def ingest_pdf(file_bytes, filename):
    metas = []
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(file_bytes.read())
    tmp.flush()

    try:
        with pdfplumber.open(tmp.name) as pdf:
            full = ""
            for pno, page in enumerate(pdf.pages, 1):
                t = page.extract_text() or ""
                if t.strip():
                    full += f"\n\n[PAGE {pno}]\n{t}"

        chunks = chunk_text(full)
        for i, c in enumerate(chunks):
            metas.append({"id": f"{filename}_{i}", "source": filename, "text": c})

    finally:
        tmp.close()

    return metas

# ----------- Simple Vector Index (No FAISS) -----------

def load_or_create_index():
    if INDEX_PATH.exists() and METAS_PATH.exists():
        embs = np.load(INDEX_PATH)
        with open(METAS_PATH, "rb") as f:
            metas = pickle.load(f)
    else:
        embs = np.zeros((0, 1536), dtype="float32")
        metas = []
    return embs, metas

embs_store, metas = load_or_create_index()

def persist_index():
    np.save(INDEX_PATH, embs_store)
    with open(METAS_PATH, "wb") as f:
        pickle.dump(metas, f)

def search_index(query, top_k=10):
    if len(embs_store) == 0:
        return []

    q_emb = embed_query(query)
    if q_emb is None:
        return []

    q_emb = q_emb.astype("float32")

    norms = np.linalg.norm(embs_store, axis=1) * np.linalg.norm(q_emb)
    sims = np.dot(embs_store, q_emb) / (norms + 1e-12)

    top_idx = np.argsort(-sims)[:top_k]
    results = [{"score": float(sims[i]), "meta": metas[i]} for i in top_idx]
    return results

# ----------- LIVE APIs -----------

def looks_like_live_query(q):
    ql = q.lower()
    if any(w in ql for w in ["weather", "temperature", "forecast"]):
        return "weather"
    if "news" in ql:
        return "news"
    return None

def call_openweather(city="New Delhi"):
    if not OPENWEATHER_KEY:
        return "No weather API key configured."
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=metric"
    try:
        r = requests.get(url, timeout=5)
        j = r.json()
        desc = j["weather"][0]["description"]
        temp = j["main"]["temp"]
        hum = j["main"]["humidity"]
        return f"Weather in {city}: {desc}, Temp={temp}°C, Humidity={hum}%"
    except Exception as e:
        return f"Weather API error: {e}"

def call_newsapi(q):
    return "News API not configured."

# ----------- LLM Generation -----------

def general_llm_generate(prompt):
    if not client:
        return "OpenAI key missing."
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        max_tokens=200,
        temperature=0.2
    )
    return resp.choices[0].message.content

def llm_generate_with_context(query, top_chunks):
    srcs = "\n\n".join([
        f"[{i+1}] SOURCE: {c['source']}\n{c['text']}"
        for i, c in enumerate(top_chunks)
    ])
    prompt = (
        "Answer ONLY using the following sources.\n"
        "If not present, respond 'I don't know'.\n\n"
        f"SOURCES:\n{srcs}\n\nQUESTION: {query}\n\n"
    )
    return general_llm_generate(prompt)

# ----------- UI -----------

st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("🔎 Hybrid RAG Chatbot — Docs + Live APIs + LLM")

col_left, col_right = st.columns([1, 2])

# ----------- LEFT PANEL -----------

with col_left:
    st.subheader("Index PDFs")
    uploaded = st.file_uploader("📎 Upload PDF", type=["pdf"])

    if st.button("Index File"):
        if not uploaded:
            st.warning("Upload a PDF first.")
        else:
            global embs_store, metas

            metas_new = ingest_pdf(uploaded, uploaded.name)
            texts = [m["text"] for m in metas_new]

            embs_new = embed_texts(texts)
            if embs_new is None:
                st.error("OPENAI_API_KEY missing — cannot create embeddings.")
            else:
                if embs_store.size == 0:
                    embs_store = embs_new
                else:
                    if embs_new.shape[1] == embs_store.shape[1]:
                        embs_store = np.vstack([embs_store, embs_new])
                    else:
                        st.error("Embedding dimension mismatch.")
                        st.stop()

                metas.extend(metas_new)
                persist_index()
                st.success(f"Indexed {len(metas_new)} chunks. Total chunks: {len(metas)}")

    if st.button("Clear Index"):
        global embs_store, metas
        embs_store = np.zeros((0, 1536), dtype="float32")
        metas = []
        if INDEX_PATH.exists(): INDEX_PATH.unlink()
        if METAS_PATH.exists(): METAS_PATH.unlink()
        st.warning("Index cleared.")

    st.write(f"Chunks stored: {len(metas)}")

# ----------- RIGHT PANEL -----------

with col_right:
    q = st.text_input("Ask anything…")

    if st.button("Ask"):
        if not q:
            st.warning("Enter a question.")
        else:
            live = looks_like_live_query(q)

            if live == "weather":
                parts = q.split("in")
                city = parts[-1].strip() if len(parts) > 1 else "New Delhi"
                st.subheader("🌤️ Weather")
                st.write(call_openweather(city))

            elif live == "news":
                st.subheader("📰 News")
                st.write(call_newsapi(q))

            else:
                results = search_index(q)
                if not results:
                    st.info("No relevant documents found — using LLM fallback.")
                    st.write(general_llm_generate(q))
                else:
                    best = results[0]
                    if best["score"] >= RAG_THRESHOLD:
                        top_chunks = [r["meta"] for r in results[:5]]
                        ans = llm_generate_with_context(q, top_chunks)
                        st.subheader("📄 Document Answer")
                        st.write(ans)

                        st.markdown("### Sources Used")
                        for i, tc in enumerate(top_chunks, 1):
                            st.write(f"{i}. {tc['source']} — {tc['text'][:200]}...")
                    else:
                        st.info("Low similarity — using general LLM.")
                        st.write(general_llm_generate(q))
