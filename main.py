import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import requests
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st

def fetch_and_extract(url):
    """Fetch a PDF from a URL and extract text."""
    response = requests.get(url)
    with open("data/temp.pdf", "wb") as f:
        f.write(response.content)
    reader = PdfReader("temp.pdf")
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text

# Example URL (replace with a real one)
url = "https://universitycounsel.ubc.ca/files/2022/05/Health-and-Safety-Policy_SC1.pdf"
document_text = fetch_and_extract(url)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(document_text)
print(f"Split into {len(chunks)} chunks")


# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = FAISS.from_texts(chunks, embeddings)
print("Vector store created")


class OpenRouterLLM(LLM):
    api_key: str  # Define as a class-level Pydantic field
    model: str = "meta-llama/llama-3.1-8b-instruct"  # Default model

    def __init__(self, api_key: str, model: str = None, **kwargs):
        # Pass api_key to the parent class (LLM) via super().__init__()
        super().__init__(api_key=api_key, **kwargs)
        if model:
            self.model = model  # Override default model if provided

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

# Initialize with your API key
api_key = "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"
llm = OpenRouterLLM(api_key=api_key)

# Fixed OpenRouterLLM class
class OpenRouterLLM(LLM):
    api_key: str
    model: str = "meta-llama/llama-3.1-8b-instruct"

    def __init__(self, api_key: str, model: str = None, **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        if model:
            self.model = model

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

# Fetch and extract
def fetch_and_extract(url):
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)
    reader = PdfReader("temp.pdf")
    return "".join(page.extract_text() or "" for page in reader.pages)

# Main app
st.title("Policy Analyzer")
url = st.text_input("Enter Policy URL", value="https://strategicplan.ubc.ca/wp-content/uploads/2019/09/2018_UBC_Strategic_Plan_Full-20180425.pdf")
query = st.text_input("Ask a Question", value="What are the strategic goals?")
if st.button("Analyze"):
    with st.spinner("Fetching and analyzing..."):
        # Prepare document
        document_text = fetch_and_extract(url)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(document_text)

        # Embed
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embeddings)

        # RAG with OpenRouter
        llm = OpenRouterLLM(api_key="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")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        result = qa_chain.run(query)
        st.write("Answer:", result)