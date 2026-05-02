import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Starting ingestion process...")
    base_dir = Path(__file__).parent
    file_path = base_dir / "mediumblog1.txt"

    # Fix proxy issues
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if os.getenv(key) in {"http://127.0.0.1:9", "https://127.0.0.1:9"}:
            os.environ.pop(key, None)

    # Load document properly
    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    print("splitting text into chunks...")

    #Chunking the text into smaller pieces for better embedding and retrieval performance
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    print(f"created {len(texts)} chunks of text")

    print("creating embeddings...")
    
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=os.getenv("NVIDIA_NV_API"),
    )

    print("uploading to Pinecone...")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name="simplerag",
        async_req=False,
    )

    print("Ingestion process completed successfully!")