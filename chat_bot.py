import streamlit as st
import tempfile
import re

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="Deekesh_GPT",
    layout="wide"
)

groq_api_key = st.secrets["GROQ_API_KEY"]

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.stApp {
    background-color: #0b1120;
}

.block-container {
    max-width: 1100px;
    padding-top: 20px;
    padding-bottom: 120px;
}

.main-title {
    text-align: center;
    color: white;
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 8px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 25px;
}

.user-msg {
    background: #22c55e;
    color: white;
    padding: 14px 18px;
    border-radius: 18px;
    margin-left: auto;
    margin-bottom: 14px;
    width: fit-content;
    max-width: 75%;
}

.bot-msg {
    background: #111827;
    color: white;
    padding: 14px 18px;
    border-radius: 18px;
    margin-right: auto;
    margin-bottom: 18px;
    width: fit-content;
    max-width: 75%;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Chat normally or upload PDF / paste YouTube link in chat</div>',
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""

model = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

for msg in st.session_state.messages:

    st.markdown(
        f'<div class="user-msg">{msg["user"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="bot-msg">{msg["bot"]}</div>',
        unsafe_allow_html=True
    )

col1, col2 = st.columns([1, 8])

with col1:
    uploaded_file = st.file_uploader(
        "",
        type=["pdf"],
        label_visibility="collapsed"
    )

with col2:
    user_input = st.chat_input("Type your message...")

if uploaded_file:

    if uploaded_file.name != st.session_state.pdf_name:

        with st.spinner("Reading PDF..."):

            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            )

            temp_file.write(uploaded_file.read())
            temp_file.close()

            loader = PyPDFLoader(temp_file.name)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.split_documents(docs)

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.pdf_name = uploaded_file.name

        st.success(f"PDF Loaded: {uploaded_file.name}")

if user_input:

    youtube_pattern = r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/\S+"

    if re.search(youtube_pattern, user_input):

        with st.spinner("Processing YouTube Video..."):

            url_match = re.search(youtube_pattern, user_input)
            youtube_url = url_match.group()

            loader = YoutubeLoader.from_youtube_url(youtube_url)
            docs = loader.load()

            raw_text = docs[0].page_content
            clean_text = re.sub(r'\s+', ' ', raw_text).strip()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = splitter.create_documents([clean_text])

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

        response_text = "YouTube video loaded successfully. Now ask your questions."

        st.session_state.messages.append({
            "user": user_input,
            "bot": response_text
        })

        st.rerun()

    st.session_state.messages.append({
        "user": user_input,
        "bot": "Thinking..."
    })

    if st.session_state.vectorstore:

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(user_input)

        context = "\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful AI assistant.

Answer using provided context when relevant.

Context:
{context}

Question:
{user_input}
"""

        response = model.invoke(prompt)

    else:
        response = model.invoke(user_input)

    st.session_state.messages[-1]["bot"] = response.content

    st.rerun()
