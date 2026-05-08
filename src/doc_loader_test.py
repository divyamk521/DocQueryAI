import time
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader

FILE_PATH=r"D:\DocQueryAI\data\Health.pdf" 

def test_loader(loader_class, name):
    start = time.time()
    loader = loader_class(FILE_PATH)
    docs = loader.load()
    end = time.time()
    print(f"{name} took: {end - start:.4f}s | Pages: {len(docs)}")
    return docs

# Compare them side-by-side
pymu_docs = test_loader(PyMuPDFLoader, "PyMuPDF")
pypdf_docs = test_loader(PyPDFLoader, "PyPDF")