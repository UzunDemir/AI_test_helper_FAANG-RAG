# app.py

import os
import streamlit as st
import requests
import tempfile
import numpy as np
import faiss
import time

from datetime import datetime
from PyPDF2 import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer, CrossEncoder

# ---------------- PAGE CONFIG ----------------

st.set_page_config(layout="wide", initial_sidebar_state="auto")

# ---------------- CSS убрал header ---------------- 

st.markdown("""
<style>

header div:nth-child(2){
display:none !important;
}

[data-testid="stHeader"]{
background:rgba(0,0,0,0);
}

.center{
display:flex;
justify-content:center;
align-items:center;
flex-direction:column;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("Описание проекта")

st.sidebar.title("TEST-passer")
st.sidebar.divider()

st.sidebar.write("""

AI ассистент для прохождения тестов.

Как работает:

1️⃣ Пользователь загружает PDF  
2️⃣ Система создаёт векторную базу знаний  
3️⃣ Используется гибридный поиск  
4️⃣ AI отвечает строго по материалам

Технологии:

• RAG архитектура  
• FAISS vector search  
• Hybrid retrieval  
• HyDE query expansion  
• Cross-encoder reranking  

""")

# ---------------- HEADER только левый слайд ----------------

st.markdown("""

<div class="center">

<img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">

<h1>TEST-passer</h1>

<h2>AI ассистент по тестам</h2>

<p>Ответы строго по учебным материалам</p>

</div>

""", unsafe_allow_html=True)

st.divider()

# ---------------- MODEL CACHE ----------------

@st.cache_resource # загружаем модель один раз
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2") # Embedder

@st.cache_resource # тоже
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2") # Reranker

embedder = load_embedder()
reranker = load_reranker()

# ---------------- API ----------------

api_key = st.secrets.get("DEEPSEEK_API_KEY")

url = "https://api.deepseek.com/v1/chat/completions"

headers = {
"Authorization": f"Bearer {api_key}",
"Content-Type": "application/json"
}

# ---------------- DATA STRUCTURES ----------------

class DocumentChunk:

    def __init__(self,text,doc,page):

        self.text=text
        self.doc=doc
        self.page=page

# ---------------- KNOWLEDGE BASE ----------------

class KnowledgeBase:

    def __init__(self):

        self.chunks=[]
        self.embeddings=[]
        self.index=None

        self.vectorizer=TfidfVectorizer()
        self.tfidf=None
        self.texts=[]

        self.files=[]

    # ---------- CHUNK ----------

    # def split_text(self,text,max_chars=1500):

    #     parts=text.split("\n\n") # по прараграфам

    #     chunks=[]
    #     current=""

    #     for p in parts:

    #         if len(current)+len(p)<max_chars:
    #             current+=p+"\n\n"
    #         else:
    #             chunks.append(current)
    #             current=p

    #     if current:
    #         chunks.append(current)

    #     return chunks


    def split_text(self,text,chunk_size=900,overlap=200):

        words = text.split()
    
        chunks=[]
        i=0
    
        while i < len(words):
    
            chunk=" ".join(words[i:i+chunk_size])
            chunks.append(chunk)
    
            i += chunk_size - overlap
    
        return chunks

    # ---------- PDF ----------

    def load_pdf(self,content,name):

        tmp=None

        try:

            with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as t:
                t.write(content)
                tmp=t.name

            reader=PdfReader(tmp)

            for i,page in enumerate(reader.pages):

                text=page.extract_text()

                if text:

                    chunks=self.split_text(text)

                    # for c in chunks:

                    #     obj=DocumentChunk(c,name,i+1)

                    #     self.chunks.append(obj)
                    #     self.texts.append(c)

                    #     emb=embedder.encode(c)

                    #     self.embeddings.append(emb)


                    embs = embedder.encode(chunks, batch_size=32)

                    for c, emb in zip(chunks, embs):
                    
                        obj = DocumentChunk(c, name, i+1)
                    
                        self.chunks.append(obj)
                        self.texts.append(c)
                        self.embeddings.append(emb)

            self.files.append(name)

            # TFIDF
            self.tfidf=self.vectorizer.fit_transform(self.texts)

            # FAISS
            vectors=np.array(self.embeddings).astype("float32")

            # dim=vectors.shape[1]

            # self.index=faiss.IndexFlatL2(dim)
            # self.index.add(vectors)

            vectors = np.array(self.embeddings).astype("float32")

            if self.index is None:
            
                dim = vectors.shape[1]
                self.index = faiss.IndexFlatL2(dim)
            
            self.index.add(vectors)

            return True

        except Exception as e:

            st.error(e)
            return False

        finally:

            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    # ---------- HYDE ----------

    def hyde(self,query):

        prompt=f"""
Write a short paragraph answering the question.

Question:
{query}

Answer:
"""

        data={
        "model":"deepseek-chat",
        "messages":[{"role":"user","content":prompt}],
        "max_tokens":200,
        "temperature":0.3
        }

        try:

            r=requests.post(url,headers=headers,json=data,timeout=30)

            if r.status_code==200:

                return r.json()['choices'][0]['message']['content']

        except:
            pass

        return query

    # ---------- SEARCH ----------

    def semantic(self,query,k=6):

        q=embedder.encode([query]).astype("float32")

        d,i=self.index.search(q,k)

        return [self.chunks[x] for x in i[0]]

    def keyword(self,query,k=6):

        q=self.vectorizer.transform([query])

        sims=cosine_similarity(q,self.tfidf)

        idx=np.argsort(sims[0])[-k:]

        return [self.chunks[i] for i in idx]

    def retrieve(self,query,k=3):

        hyp=self.hyde(query)

        search=query+" "+hyp

        s=self.semantic(search)
        k2=self.keyword(search)

        combined=s+k2

        unique=list({c.text:c for c in combined}.values())

        pairs=[[query,c.text] for c in unique]

        scores=reranker.predict(pairs)

        ranked=sorted(
            zip(unique,scores),
            key=lambda x:x[1],
            reverse=True
        )

        return [x[0] for x in ranked[:k]]

# ---------------- SESSION ----------------

if "kb" not in st.session_state:
    st.session_state.kb = KnowledgeBase()
else:
    if not hasattr(st.session_state.kb, "files"):
        st.session_state.kb = KnowledgeBase()

if "messages" not in st.session_state:
    st.session_state.messages=[]

kb=st.session_state.kb

# ---------------- FILE UPLOAD ----------------

files=st.file_uploader(
"Загрузить PDF материалы",
type="pdf",
accept_multiple_files=True
)

if files:

    for f in files:

        if f.name not in kb.files:

            ok=kb.load_pdf(f.read(),f.name)

            if ok:
                st.success(f"{f.name} загружен")

# ---------------- SHOW DOCS ----------------

if kb.files:

    st.subheader("📚 Загруженные документы")

    for d in kb.files:
        st.markdown(f"- {d}")

else:

    st.info("Документы не загружены")

# ---------------- CHAT HISTORY ----------------

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.markdown(m["content"])

# ---------------- CHAT INPUT ----------------

if prompt:=st.chat_input("Введите вопрос"):

    st.session_state.messages.append(
    {"role":"user","content":prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    #start=datetime.now()
    start=datetime.now()

    with st.spinner("🔎 AI анализирует материалы..."):
        chunks = kb.retrieve(prompt)

    #chunks=kb.retrieve(prompt)

    if not chunks:

        answer="Ответ не найден в материалах"

    else:

        context=""
        MAX_CONTEXT = 6000

        for c in chunks:

            context+=f"""
Документ: {c.doc}
Страница: {c.page}

{c.text}

"""

        full_prompt=f"""
Answer strictly based on the materials.

Question:
{prompt}

Materials:
{context}
"""

        data={
        "model":"deepseek-chat",
        "messages":[{"role":"user","content":full_prompt}],
        "temperature":0.1,
        "max_tokens":1000
        }

        # r=requests.post(url,headers=headers,json=data,timeout=60)

        # answer=r.json()['choices'][0]['message']['content']

        with st.spinner("🤖 AI формирует ответ..."):

            r=requests.post(url,headers=headers,json=data,timeout=60)
        
            answer=r.json()['choices'][0]['message']['content']

        # sources="\n\nИсточники:\n"

        # for c in chunks:
        #     sources+=f"- {c.doc}, стр {c.page}\n"

        sources="\n\nИсточники:\n"

        unique=set()
        
        for c in chunks:
        
            key=f"{c.doc}, стр {c.page}"
        
            if key not in unique:
                unique.add(key)
                sources+=f"- {key}\n"

        answer+=sources

    st.session_state.messages.append(
    {"role":"assistant","content":answer}
    )

    with st.chat_message("assistant"):
        st.markdown(answer)

    end=datetime.now()

    st.info(
    f"⏱️ Ответ найден за {(end-start).total_seconds():.2f} сек"
    )

# ---------------- CLEAR ----------------

if st.button("Очистить чат"):

    st.session_state.messages=[]
    st.rerun()
