import streamlit as st
from pathlib import Path
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os

st.set_page_config(page_title="SPEC/MEMO Q&A", layout="wide")
st.title("📄 Space SPEC/MEMO Q&A System (Local Simulation)")

# Create persistent vectorstore folder (in-memory otherwise resets on refresh)
VECTOR_DIR = "vector_db"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Load PDF documents from content folder
@st.cache_resource
def load_and_embed_documents():
    st.info("🔄 Loading and embedding documents...")

    files = Path("content").glob("*.pdf")
    all_docs = []
    for file in files:
        loader = PyMuPDFLoader(str(file))
        all_docs.extend(loader.load())

    if not all_docs:
        st.warning("No documents found in content folder.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = splitter.split_documents(all_docs)


    from langchain.embeddings import FakeEmbeddings
    embeddings = FakeEmbeddings(size=768)

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTOR_DIR)
    vectorstore.persist()

    st.success(f"✅ Loaded {len(all_docs)} documents and created {len(splits)} chunks.")
    return vectorstore

# Load or initialize vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_and_embed_documents()

# Ask a question
st.header("💬 Ask a question about your documents")

query = st.text_input("🔎 Your question:", placeholder="e.g. What is the battery capacity?")
if query and st.session_state.vectorstore:
    results = st.session_state.vectorstore.similarity_search(query, k=3)

    if results:
        st.subheader("🔍 Top Matches:")
        for i, doc in enumerate(results):
            st.markdown(f"**Match {i+1}:**")
            st.write(doc.page_content[:1000])
    else:
        st.warning("No results found for your question.")