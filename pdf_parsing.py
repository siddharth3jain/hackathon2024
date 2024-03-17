import os
from pathlib import Path
from parameters import dataset_dir, root_dir, chroma_persist_directory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from initialization import embeddings_function


def create_chunks_db_all():
    documents = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(dataset_dir, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    # save to disk
    Chroma.from_documents(chunked_documents, embeddings_function, persist_directory=str(Path(chroma_persist_directory, 'combined')))
    print("Vector db created and stored")
    return


def create_chunks_db_individual():
    for file in os.listdir(dataset_dir):
        count = 0
        if file.endswith('.pdf'):
            pdf_path = os.path.join(dataset_dir, file)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunked_documents = text_splitter.split_documents(documents)
            # save to disk
            Chroma.from_documents(chunked_documents, embeddings_function, persist_directory=str(os.path.join(chroma_persist_directory, 'individual', file)))
            print(f"Vector db created and stored for - {file}")
            count = count + 1
        if count ==1:
            break
    return


if __name__ == "__main__":
    create_chunks_db_all()
    # create_chunks_db_individual()
