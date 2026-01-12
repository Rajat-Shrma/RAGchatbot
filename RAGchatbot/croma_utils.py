from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

import tempfile
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)
from typing import List
from CONSTS import vectorstore
def load_documents(file)-> List[Document]:
    suffix = file.filename.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    if suffix == "txt":
        loader = TextLoader(tmp_path, encoding="utf-8")

    elif suffix == "pdf":
        loader = PyPDFLoader(tmp_path)

    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    return documents

def splitting_documents(docs: List[Document])-> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 700,
        chunk_overlap = 150,
        separators = ['']
    )

    texts = text_splitter.split_documents(docs)

    return texts

def index_document_to_chroma(file: object , file_id: int)->bool:
    try:
        docs = load_documents(file)
        splits = splitting_documents(docs)

        for split in splits:
            split.metadata['file_id'] = file_id
        
        vectorstore.add_documents(splits)

        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False
    

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False