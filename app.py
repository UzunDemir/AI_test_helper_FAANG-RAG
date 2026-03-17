import os
import streamlit as st
import requests
import tempfile
import numpy as np
import faiss
from datetime import datetime
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from functools import lru_cache

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", initial_sidebar_state="auto")

# ---------------- CSS ----------------
st.markdown("""
<style>
header div:nth-child(2){ display:none !important; }
[data-testid="stHeader"]{ background:rgba(0,0,0,0); }
.center{
    display:flex;
    justify-content:center;
    align-items:center;
    flex-direction:column;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("AI Pipeline")
st.sidebar.write("""
Features:
• Self-RAG  
• Query Routing  
• Hybrid Retrieval  
• HyDE Expansion  
• Multi Query  
• Cross-Encoder Rerank  
• Memory Retrieval  
""")
st.sidebar.markdown("---")

# ---------------- HEADER ----------------
st.markdown("""
<div class="center">
<img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">
<h1>TEST-passer</h1>
<h2>AI ассистент по тестам</h2>
<p>Ответы строго по учебным материалам</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------- MODELS ----------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

embedder = load_embedder()
reranker = load_reranker()

# ---------- API ----------
api_key = st.secrets.get("DEEPSEEK_API_KEY")
url = "https://api.deepseek.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

# ---------- MEMORY ----------
class ConversationMemory:
    def __init__(self, max_messages=50):
        self.messages = []
        self.embeddings = None
        self.max_messages = max_messages

    def add(self, text):
        if not text or not text.strip():
            return
        emb = embedder.encode([text])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.messages.append(text)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
            self.embeddings = np.vstack([embedder.encode(self.messages)])

    def search(self, query, k=3):
        if self.embeddings is None or not self.messages:
            return []
        q = embedder.encode([query])
        sims = cosine_similarity(q, self.embeddings)[0]
        idx = np.argsort(sims)[-k:]
        return [self.messages[i] for i in idx if self.messages[i].strip()]

# ---------- KNOWLEDGE BASE ----------
class KnowledgeBase:
    def __init__(self, nlist=100):
        self.chunks = []  # tuple: (text, file, page)
        self.texts = []   # для векторов и TF-IDF
        self.files = []
        self.index = None
        self.vectorizer = TfidfVectorizer()
        self.tfidf = None
        self.nlist = nlist

    def split_text(self, text, chunk_size=800, overlap=150):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def load_pdf(self, data, name):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                t.write(data)
                tmp = t.name

            reader = PdfReader(tmp)
            new_chunks, new_texts = [], []

            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    for c in self.split_text(text):
                        new_chunks.append((c, name, page_num))  # сохраняем источник и страницу
                        new_texts.append(c)

            if not new_chunks:
                return False

            emb = embedder.encode(new_texts)
            vectors = np.array(emb).astype("float32")
            dim = vectors.shape[1]

            # SAFE FAISS
            if len(vectors) < 5:
                if self.index is None:
                    self.index = faiss.IndexFlatL2(dim)
                self.index.add(vectors)
            else:
                nlist_train = min(self.nlist, len(vectors))
                if self.index is None:
                    quantizer = faiss.IndexFlatL2(dim)
                    self.index = faiss.IndexIVFFlat(quantizer, dim, nlist_train, faiss.METRIC_L2)
                    self.index.train(vectors)
                elif not self.index.is_trained:
                    self.index.train(vectors)
                self.index.add(vectors)

            self.chunks += new_chunks
            self.texts += new_texts
            self.tfidf = self.vectorizer.fit_transform(self.texts)
            self.files.append(name)

            return True
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    def semantic(self, query, k=5):
        if self.index is None or not self.chunks:
            return []
        v = embedder.encode([query]).astype("float32")
        d, i = self.index.search(v, k)
        result = []
        for idx in i[0]:
            if idx < len(self.chunks):
                _, file, page = self.chunks[idx]
                result.append(f"[{file} | page {page}]")
        return result

    def keyword(self, query, k=5):
        if self.tfidf is None or not self.chunks:
            return []
        vec = self.vectorizer.transform([query])
        sims = cosine_similarity(vec, self.tfidf)[0]
        idx = np.argsort(sims)[-k:]
        result = []
        for i in idx:
            if i < len(self.chunks):
                _, file, page = self.chunks[i]
                result.append(f"[{file} | page {page}]")
        return result

    # def retrieve(self, query, top_k=4):
    #     sem = self.semantic(query)
    #     key = self.keyword(query)
    #     unique = list({t for t in sem + key if t.strip()})
    #     if not unique:
    #         return []
    #     pairs = [[query, t] for t in unique]
    #     scores = reranker.predict(pairs)
    #     ranked = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)
    #     st.session_state.reranker_log = [(text, float(score)) for text, score in ranked]
    #     return [x[0] for x in ranked[:top_k]]
####################################################################


def retrieve(self, query, top_k=4):
    sem = self.semantic(query)
    key = self.keyword(query)
    unique_chunks = list({t for t in sem + key if t.strip()})
    if not unique_chunks:
        return []
    pairs = [[query, t] for t in unique_chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(unique_chunks, scores), key=lambda x: x[1], reverse=True)
    st.session_state.reranker_log = [(t[:50]+"...", float(s)) for t, s in ranked]
    
    # вернем список вида: ("текст", "file.pdf | page X")
    result = []
    for t, _ in ranked[:top_k]:
        for chunk_text, file, page in self.chunks:
            if t == chunk_text:
                result.append((t, f"{file} | page {page}"))
                break
    return result

# ---------- SELF-RAG ----------
@lru_cache(maxsize=128)
def self_rag(query):
    prompt = f"""
Decide if the question requires document retrieval.

Question:
{query}

Answer ONLY:
SEARCH
or
NO_SEARCH
"""
    data = {"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"max_tokens":10}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        res = r.json()['choices'][0]['message']['content']
        return "SEARCH" in res.upper()
    except:
        return True

# ---------- QUERY ROUTER ----------
@lru_cache(maxsize=128)
def route_query(query):
    prompt = f"""
Classify the query.

Options:
DOCUMENT
CONVERSATION
GENERAL

Query:
{query}

Answer with one word.
"""
    data = {"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"max_tokens":10}
    try:
        r = requests.post(url, headers=headers, json=data, timeout=15)
        return r.json()['choices'][0]['message']['content'].strip().upper()
    except:
        return "DOCUMENT"

# ---------- SESSION ----------
if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase(nlist=200)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_messages=50)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "reranker_log" not in st.session_state:
    st.session_state.reranker_log = []

kb = st.session_state.kb
memory = st.session_state.memory

# ---------- FILE UPLOAD ----------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if files:
    for f in files:
        if f.name not in kb.files:
            with st.spinner(f"Processing {f.name}..."):
                kb.load_pdf(f.read(), f.name)
            st.success(f"{f.name} added")

# ---------- CHAT HISTORY ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- CHAT ----------
if prompt := st.chat_input("Ask question"):
    st.session_state.messages.append({"role":"user","content":prompt})
    memory.add(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    start = datetime.now()
    route = route_query(prompt)
    context_text = ""

    if route == "DOCUMENT" and self_rag(prompt):
        with st.spinner("Searching documents..."):
            context_chunks = kb.retrieve(prompt)  # список tuple: (text, "[file | page]")
            if context_chunks:
                # формируем строку с текстом и источником для LLM
                context_text = "\n".join([f"{t} ({fp})" for t, fp in context_chunks])
            else:
                context_text = ""
    elif route == "CONVERSATION":
        mem = memory.search(prompt)
        if mem:
            context_text = "\n".join(mem)
        else:
            context_text = ""
    
    llm_prompt = f"""
Answer the question.

Context (document name and page only):
{context_text}

Question:
{prompt}
"""
    data = {"model":"deepseek-chat","messages":[{"role":"user","content":llm_prompt}],"temperature":0.1}

    with st.spinner("AI thinking..."):
        try:
            r = requests.post(url, headers=headers, json=data, timeout=60)
            answer = r.json()['choices'][0]['message']['content']
        except:
            answer = "⚠️ AI server error or connection failed."

    memory.add(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        for para in answer.split("\n"):
            placeholder.markdown(para)
    
    end = datetime.now()
    st.info(f"⏱ {(end-start).total_seconds():.2f} sec")

# ---------- DEBUG LOG ----------
if st.sidebar.checkbox("Show reranker log"):
    st.sidebar.write(st.session_state.reranker_log)

# ---------- CLEAR ----------
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
if st.button("Remove documents"):
    st.session_state.kb = KnowledgeBase(nlist=200)
    st.session_state.memory = ConversationMemory(max_messages=50)
    st.rerun()
