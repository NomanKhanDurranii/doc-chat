import os
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"

def load_all_documents(data_path):
    documents = []

    # Load PDF files
    pdf_loader = PyPDFLoader
    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(data_path, file)
            loader = pdf_loader(file_path)
            documents.extend(loader.load())

    # Load CSV files
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            file_path = os.path.join(data_path, file)
            loader = CSVLoader(file_path=file_path)
            documents.extend(loader.load())

    # Load Excel files
    for file in os.listdir(data_path):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(data_path, file)
            loader = UnstructuredExcelLoader(file_path)
            documents.extend(loader.load())

    return documents

# Load all documents
documents = load_all_documents(DATA_PATH)

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Generate embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(chunks, embedding_model)
db.save_local(DB_FAISS_PATH)


