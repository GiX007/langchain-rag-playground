"""Load a .txt and a .pdf file with LangChain loaders and show the Document objects (content and metadata).

Run with: uv run src/langchain_rag_playground/loading/document_loaders.py
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_text_file(file_path: str) -> None:
    """Load a text file and print its content and metadata."""
    loader = TextLoader(file_path)
    documents = loader.load()
    print(f"\nDocument object type: {type(documents)}")
    print(f"Document object: {documents}")

    print(f"Loaded {len(documents)} document(s) from {file_path}")
    print(f"First document content: {documents[0].page_content[:100]}...")  # Print first 100 characters
    print(f"First document metadata: {documents[0].metadata}")


def load_pdf_file(file_path: str) -> None:
    """Load a PDF file and print the content and metadata of each page."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"\nDocument object type: {type(documents)}")
    print(f"Document object: {documents}")

    print(f"Loaded {len(documents)} document(s) from {file_path}")

    # PyPDFLoader returns one Document per page
    for i, doc in enumerate(documents):
        print(f"\nDocument (Page) {i + 1} content: {doc.page_content[:100]}...")  # Print first 100 characters
        print(f"\nDocument (Page) {i + 1} metadata: {doc.metadata}")

if __name__ == "__main__":
    # load_text_file("docs/test1.txt")  # Load one text file from the docs directory
    load_pdf_file("docs/langchain_document_loaders.pdf")  # Load one PDF file from the docs directory