# import os
# import streamlit as st
# import requests
# import tempfile
# import numpy as np
# import faiss

# from datetime import datetime
# from PyPDF2 import PdfReader

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from sentence_transformers import SentenceTransformer, CrossEncoder


# # ---------------- PAGE CONFIG ----------------

# st.set_page_config(layout="wide", initial_sidebar_state="auto")

# # ---------------- CSS ----------------

# st.markdown("""
# <style>

# header div:nth-child(2){
# display:none !important;
# }

# [data-testid="stHeader"]{
# background:rgba(0,0,0,0);
# }

# .center{
# display:flex;
# justify-content:center;
# align-items:center;
# flex-direction:column;
# text-align:center;
# }

# </style>
# """, unsafe_allow_html=True)


# # ---------------- SIDEBAR ----------------

# st.sidebar.title("TEST-passer")
# st.sidebar.divider()

# st.sidebar.write("""
# AI ассистент для прохождения тестов.

# Как работает:

# 1️⃣ Пользователь загружает PDF  
# 2️⃣ Создается векторная база знаний  
# 3️⃣ Используется гибридный поиск  
# 4️⃣ AI отвечает только по материалам

# Технологии:

# • RAG architecture  
# • FAISS vector search  
# • Hybrid retrieval  
# • HyDE query expansion  
# • Multi-query retrieval  
# • Cross-encoder reranking  
# • Answer verification  
# """)


# # ---------------- HEADER ----------------

# st.markdown("""
# <div class="center">

# <img src="https://github.com/UzunDemir/mnist_777/blob/main/200w.gif?raw=true">

# <h1>TEST-passer</h1>
# <h2>AI ассистент по тестам</h2>

# <p>Ответы строго по учебным материалам</p>

# </div>
# """, unsafe_allow_html=True)

# st.divider()


# # ---------------- MODEL CACHE ----------------

# @st.cache_resource
# def load_embedder():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_resource
# def load_reranker():
#     return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# embedder = load_embedder()
# reranker = load_reranker()


# # ---------------- API ----------------

# api_key = st.secrets.get("DEEPSEEK_API_KEY")

# url = "https://api.deepseek.com/v1/chat/completions"

# headers = {
# "Authorization": f"Bearer {api_key}",
# "Content-Type": "application/json"
# }


# # ---------------- DATA STRUCTURES ----------------

# class DocumentChunk:

#     def __init__(self,text,doc,page):

#         self.text=text
#         self.doc=doc
#         self.page=page


# # ---------------- KNOWLEDGE BASE ----------------

# class KnowledgeBase:

#     def __init__(self):

#         self.chunks=[]
#         self.index=None

#         self.vectorizer=TfidfVectorizer()
#         self.tfidf=None

#         self.texts=[]
#         self.files=[]

#     # ---------- CHUNKING ----------

#     def split_text(self,text,chunk_size=900,overlap=200):

#         words=text.split()

#         chunks=[]
#         i=0

#         while i < len(words):

#             chunk=" ".join(words[i:i+chunk_size])
#             chunks.append(chunk)

#             i += chunk_size-overlap

#         return chunks


#     # ---------- PDF LOADER ----------

#     def load_pdf(self,content,name):

#         tmp=None

#         try:

#             with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as t:
#                 t.write(content)
#                 tmp=t.name

#             reader=PdfReader(tmp)

#             new_chunks=[]
#             new_texts=[]

#             for i,page in enumerate(reader.pages):

#                 text=page.extract_text()

#                 if text:

#                     chunks=self.split_text(text)

#                     for c in chunks:

#                         new_chunks.append(DocumentChunk(c,name,i+1))
#                         new_texts.append(c)

#             if not new_chunks:
#                 return False

#             embeddings=embedder.encode(new_texts,batch_size=32)

#             vectors=np.array(embeddings).astype("float32")

#             if self.index is None:

#                 dim=vectors.shape[1]
#                 self.index=faiss.IndexFlatL2(dim)

#             self.index.add(vectors)

#             self.chunks.extend(new_chunks)
#             self.texts.extend(new_texts)

#             self.tfidf=self.vectorizer.fit_transform(self.texts)

#             self.files.append(name)

#             return True

#         finally:

#             if tmp and os.path.exists(tmp):
#                 os.remove(tmp)


#     # ---------- HYDE ----------

#     def hyde(self,query):

#         prompt=f"""
# Write a short paragraph answering the question.

# Question:
# {query}
# """

#         data={
#         "model":"deepseek-chat",
#         "messages":[{"role":"user","content":prompt}],
#         "max_tokens":150
#         }

#         try:

#             r=requests.post(url,headers=headers,json=data,timeout=30)

#             if r.status_code==200:
#                 return r.json()['choices'][0]['message']['content']

#         except:
#             pass

#         return query


#     # ---------- MULTI QUERY ----------

#     def multi_query(self,query):

#         prompt=f"""
# Generate 3 search queries for the question.

# Question:
# {query}
# """

#         queries=[query]

#         data={
#         "model":"deepseek-chat",
#         "messages":[{"role":"user","content":prompt}],
#         "max_tokens":120
#         }

#         try:

#             r=requests.post(url,headers=headers,json=data,timeout=30)

#             if r.status_code==200:

#                 text=r.json()['choices'][0]['message']['content']

#                 for q in text.split("\n"):

#                     q=q.strip("- ").strip()

#                     if len(q)>5:
#                         queries.append(q)

#         except:
#             pass

#         return queries[:4]


#     # ---------- SEARCH ----------

#     def semantic(self,query,k=6):

#         if self.index is None:
#             return []

#         q=embedder.encode([query]).astype("float32")

#         d,i=self.index.search(q,k)

#         results=[]

#         for idx in i[0]:

#             if idx < len(self.chunks):
#                 results.append(self.chunks[idx])

#         return results


#     def keyword(self,query,k=6):

#         if self.tfidf is None:
#             return []

#         q=self.vectorizer.transform([query])

#         sims=cosine_similarity(q,self.tfidf)

#         idx=np.argsort(sims[0])[-k:]

#         return [self.chunks[i] for i in idx]


#     # ---------- RETRIEVE ----------

#     def retrieve(self,query,k=3):

#         queries=self.multi_query(query)

#         all_chunks=[]

#         for q in queries:

#             hyp=self.hyde(q)

#             search=q+" "+hyp

#             all_chunks+=self.semantic(search)
#             all_chunks+=self.keyword(search)

#         unique=list({c.text:c for c in all_chunks}.values())

#         if not unique:
#             return []

#         pairs=[[query,c.text] for c in unique]

#         scores=reranker.predict(pairs)

#         ranked=sorted(zip(unique,scores),key=lambda x:x[1],reverse=True)

#         return [x[0] for x in ranked[:k]]


#     # ---------- VERIFY ----------

#     def verify_answer(self,question,answer,context):

#         prompt=f"""
# Check if the answer is supported by the materials.

# Question:
# {question}

# Answer:
# {answer}

# Materials:
# {context}

# Reply only:

# SUPPORTED
# or
# NOT_SUPPORTED
# """

#         data={
#         "model":"deepseek-chat",
#         "messages":[{"role":"user","content":prompt}],
#         "max_tokens":50,
#         "temperature":0
#         }

#         try:

#             r=requests.post(url,headers=headers,json=data,timeout=30)

#             if r.status_code==200:

#                 res=r.json()['choices'][0]['message']['content']

#                 if "SUPPORTED" in res:
#                     return True

#         except:
#             pass

#         return False


# # ---------------- SESSION ----------------

# if "kb" not in st.session_state:
#     st.session_state.kb=KnowledgeBase()

# if "messages" not in st.session_state:
#     st.session_state.messages=[]

# kb=st.session_state.kb


# # ---------------- FILE UPLOAD ----------------

# files=st.file_uploader(
# "Загрузить PDF материалы",
# type="pdf",
# accept_multiple_files=True
# )

# if files:

#     for f in files:

#         if f.name not in kb.files:

#             ok=kb.load_pdf(f.read(),f.name)

#             if ok:
#                 st.success(f"{f.name} загружен")


# # ---------------- SHOW DOCS ----------------

# if kb.files:

#     st.subheader("📚 Загруженные документы")

#     for d in kb.files:
#         st.markdown(f"- {d}")

# else:
#     st.info("Документы не загружены")


# # ---------------- CHAT HISTORY ----------------

# for m in st.session_state.messages:

#     with st.chat_message(m["role"]):
#         st.markdown(m["content"])


# # ---------------- CHAT INPUT ----------------

# if prompt:=st.chat_input("Введите вопрос"):

#     st.session_state.messages.append(
#     {"role":"user","content":prompt}
#     )

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     start=datetime.now()

#     with st.spinner("🔎 AI ищет информацию..."):

#         chunks=kb.retrieve(prompt)

#     if not chunks:

#         answer="Ответ не найден в материалах."

#     else:

#         context=""

#         for c in chunks:

#             context+=f"""
# Документ: {c.doc}
# Страница: {c.page}

# {c.text}
# """

#         full_prompt=f"""
# Answer strictly using the materials.

# Question:
# {prompt}

# Materials:
# {context}
# """

#         data={
#         "model":"deepseek-chat",
#         "messages":[{"role":"user","content":full_prompt}],
#         "max_tokens":1000,
#         "temperature":0.1
#         }

#         with st.spinner("🤖 AI формирует ответ..."):

#             try:

#                 r=requests.post(url,headers=headers,json=data,timeout=60)

#                 if r.status_code==200:

#                     answer=r.json()['choices'][0]['message']['content']

#                 else:
#                     answer="Ошибка AI сервера."

#             except:
#                 answer="Ошибка соединения."


#         if not kb.verify_answer(prompt,answer,context):

#             answer+="\n\n⚠️ Ответ может быть неполностью подтвержден материалами."


#         sources="\n\nИсточники:\n"

#         unique=set()

#         for c in chunks:

#             key=f"{c.doc}, стр {c.page}"

#             if key not in unique:
#                 unique.add(key)
#                 sources+=f"- {key}\n"

#         answer+=sources


#     st.session_state.messages.append(
#     {"role":"assistant","content":answer}
#     )

#     with st.chat_message("assistant"):

#         placeholder=st.empty()

#         streamed=""

#         for word in answer.split():

#             streamed+=word+" "

#             placeholder.markdown(streamed)

#         st.markdown(answer)

#     end=datetime.now()

#     st.info(f"⏱️ Ответ найден за {(end-start).total_seconds():.2f} сек")


# # ---------------- CLEAR ----------------

# if st.button("Очистить чат"):

#     st.session_state.messages=[]
#     st.rerun()


# if st.button("Удалить документы"):

#     st.session_state.kb=KnowledgeBase()
#     st.rerun()


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

# ---------------- PAGE CONFIG ----------------

st.set_page_config(layout="wide", initial_sidebar_state="auto")

# ---------------- CSS ----------------

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
    def __init__(self):
        self.messages = []
        self.embeddings = None

    def add(self, text):
        if not text or not text.strip():
            return
        emb = embedder.encode([text])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.messages.append(text)

    def search(self, query, k=3):
        if self.embeddings is None:
            return []
        q = embedder.encode([query])
        sims = cosine_similarity(q, self.embeddings)[0]
        idx = np.argsort(sims)[-k:]
        return [self.messages[i] for i in idx if self.messages[i] and self.messages[i].strip()]

# ---------- KNOWLEDGE BASE ----------
class KnowledgeBase:
    def __init__(self):
        self.chunks = []
        self.texts = []
        self.files = []
        self.index = None
        self.vectorizer = TfidfVectorizer()
        self.tfidf = None

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
            new_chunks = []
            new_texts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    for c in self.split_text(text):
                        new_chunks.append(c)
                        new_texts.append(c)
            if not new_chunks:
                return False
            emb = embedder.encode(new_texts)
            vectors = np.array(emb).astype("float32")
            if self.index is None:
                dim = vectors.shape[1]
                self.index = faiss.IndexFlatL2(dim)
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
        if self.index is None:
            return []
        v = embedder.encode([query]).astype("float32")
        d, i = self.index.search(v, k)
        result = []
        for idx in i[0]:
            if idx < len(self.chunks) and self.chunks[idx] and self.chunks[idx].strip():
                result.append(self.chunks[idx])
        return result

    def keyword(self, query, k=5):
        if self.tfidf is None:
            return []
        vec = self.vectorizer.transform([query])
        sims = cosine_similarity(vec, self.tfidf)[0]
        idx = np.argsort(sims)[-k:]
        result = []
        for i in idx:
            if i < len(self.chunks) and self.chunks[i] and self.chunks[i].strip():
                result.append(self.chunks[i])
        return result

    def retrieve(self, query, top_k=4):
        sem = self.semantic(query)
        key = self.keyword(query)
        unique = list({t for t in sem + key if isinstance(t, str) and t.strip()})
        if not unique:
            return []
        pairs = [[query, t] for t in unique]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(unique, scores), key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked[:top_k]]

# ---------- SELF-RAG ----------
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
    st.session_state.kb = KnowledgeBase()
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []

kb = st.session_state.kb
memory = st.session_state.memory

# ---------- FILE UPLOAD ----------
files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if files:
    for f in files:
        if f.name not in kb.files:
            with st.spinner("Processing PDF..."):
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
    context = ""
    if route == "DOCUMENT" and self_rag(prompt):
        with st.spinner("Searching documents..."):
            docs = kb.retrieve(prompt)
            if docs:
                docs = [d for d in docs if isinstance(d, str) and d.strip()]
                context += "\n".join(docs)
    elif route == "CONVERSATION":
        mem = memory.search(prompt)
        if mem:
            context += "\n".join(mem)
    llm_prompt = f"Answer the question.\n\nContext:\n{context}\n\nQuestion:\n{prompt}"
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
        txt = ""
        for w in answer.split():
            txt += w + " "
            placeholder.markdown(txt)
    end = datetime.now()
    st.info(f"⏱ {(end-start).total_seconds():.2f} sec")

# ---------- CLEAR ----------
if st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
if st.button("Remove documents"):
    st.session_state.kb = KnowledgeBase()
    st.session_state.memory = ConversationMemory()
    st.rerun()
