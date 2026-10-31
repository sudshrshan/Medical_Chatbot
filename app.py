
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from your .env file")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# Load HuggingFace embeddings
embeddings = download_hugging_face_embeddings()

# Connect to existing Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Initialize Groq model
chatModel = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.4
)

# Create RAG prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create document chain
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

# Create retrieval chain
rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg", "").strip()

        if not msg:
            return "Please enter a message.", 400

        print("User:", msg)

        response = rag_chain.invoke({"input": msg})

        answer = response.get("answer", "Sorry, I couldn't generate a response.")

        print("Bot:", answer)

        return str(answer)

    except Exception as e:
        app.logger.exception("Chat error")
        return "An error occurred while processing your question.", 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
