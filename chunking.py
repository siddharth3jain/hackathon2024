from langchain.document_loaders import PyPDFLoader
from tqdm import tqdm
from langchain.vectorstores import Chroma
from pathlib import Path
from langchain_text_splitters import CharacterTextSplitter
from parameters import chroma_persist_directory, dataset_dir
import os
from initialization import embeddings_function


def textsplitter(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False)

    return text_splitter.split_documents(documents)


def pdfparser(pdf_folder_path):
    loaders = [PyPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    documents = []
    for loader in tqdm(loaders):
        try:
            documents.extend(loader.load())
        except:
            pass
    return documents


def vectorstore(pdf_folder_path):
    persist_directory = Path(chroma_persist_directory,'combined')
    [f.unlink() for f in persist_directory.glob("*") if f.is_file()]

    docs_list = pdfparser(pdf_folder_path)
    chunk_list = textsplitter(docs_list)
    vectordb = Chroma.from_documents(
        documents=chunk_list,
        embedding=embeddings_function,
        persist_directory=str(persist_directory)
    )
    print("Done")
    return vectordb


if __name__=="__main__":
    vectorstore(dataset_dir)
