from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# insurance documents
documents = {
    "care": r"C:\Users\Rifah\rag_project\data\healthpolicy.pdf",
    "star": r"C:\Users\Rifah\rag_project\data\starpolicy.pdf",
    "niva": r"C:\Users\Rifah\rag_project\data\nivapolicy.pdf",
    "hdfc": r"C:\Users\Rifah\rag_project\data\hdfcpolicy.pdf"
}

for provider, pdf_path in documents.items():

    print(f"\nProcessing {provider} document...")

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    chunks = splitter.split_text(text)

    print(f"Chunks created: {len(chunks)}")

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=f"chroma_{provider}"
    )

    print(f"{provider} vector DB created successfully!")