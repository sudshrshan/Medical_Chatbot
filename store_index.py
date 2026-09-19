from dotenv import load_dotenv
import os
import time
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)


# Create an empty vectorstore bound to the index (doesn't upsert yet)
docsearch = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
)

def add_documents_with_retry(vectorstore, documents, batch_size=50, max_retries=5):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                vectorstore.add_documents(batch)
                print(f"Upserted batch {i} to {i + len(batch)}")
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"Batch {i} failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise Exception(f"Failed to upsert batch starting at index {i} after {max_retries} retries")

# Only run this once — re-running will re-upsert duplicates
# add_documents_with_retry(docsearch, text_chunks, batch_size=50)