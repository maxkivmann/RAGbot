from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import DirectoryLoader

def create_vectordb():
    
    loader = DirectoryLoader("./upladed_files")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss")
    return db

if __name__ == "__main__":
    create_vectordb()

