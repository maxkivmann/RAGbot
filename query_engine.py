from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import DirectoryLoader

# Function to process PDF and DOCX files
def processing_pdf_docx_files(folder_path):
    loader = DirectoryLoader(folder_path)
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


db = processing_pdf_docx_files("./files")


def dbsearch(query:str):
    """
    Search vector database and returns most relevant chunks from most relevant 'k' documents 
    """
    # Processing PDF and DOCX files

    retrieved_docs = db.similarity_search(query, k=5)
   

    plain_texts = [f"\nDocument {i + 1}:\n{doc.page_content}" for i, doc in enumerate( retrieved_docs)]
    sources = [f"\nSource of Document {i + 1}:\n {doc.metadata}" for i, doc in  enumerate( retrieved_docs)]

    # Concatenate plain texts and sources into formatted strings
    texts = '\n'.join(plain_texts)
    text_sources = '\n'.join(sources)

    return texts, text_sources, retrieved_docs

if __name__ == "__main__":
    prompt = input("query:")
    texts, text_sources, retrieved_docs = dbsearch(prompt)

    # Print or use the stored results
    print(texts)
    
    print("\nSources:")
    print(text_sources)