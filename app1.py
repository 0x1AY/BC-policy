import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore  # Updated import
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import streamlit as st
from typing import Optional, List, Mapping, Any
from pinecone import Pinecone, ServerlessSpec
import os

# Set Pinecone API key in environment
PINECONE_API_KEY = "pcsk_494APJ_9FSfXN2287oQWrgW375TkeuyPhMJm6N5xKsTvBWYbks1oPvjS9RJLcruqDAkr4d"
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
INDEX_NAME = "bc-policy-db"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists, create if not
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Matches all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Fetch and extract text from a URL
def fetch_and_extract(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)
        reader = PdfReader("temp.pdf")
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

# List of BC policy URLs
bc_policy_urls = [
    "https://universitycounsel.ubc.ca/files/2022/05/Research-Policy_LR2.pdf",
    "https://universitycounsel.ubc.ca/files/2022/05/Research-Over-Expenditure-Policy_FM4.pdf",
    "https://universitycounsel.ubc.ca/files/2022/05/Contract-Employees-Fund-Policy_FM7.pdf",
    "https://universitycounsel.ubc.ca/files/2024/11/Financial-Investigations-Policy_SC15-Consultation-Draft.pdf",
    "https://universitycounsel.ubc.ca/files/2022/05/Financial-Aid-Policy_LR10.pdf",
    "https://www.sfu.ca/content/dam/sfu/finance/publications-news/publications/budgetbook/2024-25%20SFU%20Budget_Final%20Apr%2012.pdf",
    "https://www.sfu.ca/content/dam/sfu/finance/publications-news/publications/annualreport/financial_report_2024_v16.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/10_series/A10-01.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/10_series/A10-02.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/10_series/A10-03.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/10_series/A10-06.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/12_series/A12-04.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/13_series/A13-03.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/academic_policies/32_series/A32.01%20Policy%20-%20Awards%20for%20Excellence%20in%20Teaching%20-%2001sept22.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/administrative_policies/3_series/AD3-10.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/administrative_policies/9_series/AD9-01.pdf",
    "https://www.sfu.ca/content/dam/sfu/policies/files/administrative_policies/9_series/AD9-09.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs/governance/A19%20Bullying%20and%20Harassment%20Prevention%20and%20Response_0.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs//Academic%20Integrity%20Policy.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs//Academic%20Performance%20Policy.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs//English%20Language%20Competency%20Standards%20Policy.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs/finance-dates-and-deadlines/Grading%20Policy%20May%202019.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs//A62%20Investment%20Policy.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs/governance/a20-student-non-academic-misconduct-policy.pdf",
    "https://www.douglascollege.ca/sites/default/files/docs/finance/consolidated-financial-statements-march-31-2024.pdf",
    "https://vpfo-finance-2024.sites.olt.ubc.ca/files/2024/11/2024_25_UBCBudgetReport.pdf?_gl=1*egicjh*_ga*NzgzMzAyODU5LjE3NDI5Mjk2MTI.*_ga_3B1R282RNR*MTc0MjkzNjM3My4xLjAuMTc0MjkzNjM3My4wLjAuMA..",
    "https://www.uvic.ca/budget/_assets/docs/framework/planning-budget-framework-2025.pdf",
    "https://www.uvic.ca/universitysecretary/assets/docs/policies/HR6100_1100_.pdf",
    "https://www.uvic.ca/universitysecretary/assets/docs/policies/GV0200_1105_.pdf",
    "https://www.uvic.ca/universitysecretary/assets/docs/policies/HR6115_1110_.pdf",
    "https://www.uvic.ca/universitysecretary/assets/docs/policies/GV0205_1150_.pdf",
    "https://www.uvic.ca/universitysecretary/assets/docs/policies/AC1205_2340.pdf",
    "https://www.bcit.ca/files/financialservices/pdf/financial-statements-2024.pdf",
    "https://www.bcit.ca/files/respect/5315_rdi_annual_report_final_remediated.pdf",
    "https://langara.ca/departments/financial-services/pdfs/2025-operating-and-capital-budget.pdf",
    "https://langara.ca/departments/financial-services/pdfs/2024-annual-report.pdf",
    "https://www.capilanou.ca/media/capilanouca/about-capu/governance/strategic-plan/Envisioning-2030-Capilano-University-Strategic-Plan-2023-2028.pdf",
    "https://www.tru.ca/__shared/assets/Strategic_Priorities_2022-2027_59936.pdf",
    "https://www.kpu.ca/sites/default/files/Vision%202026%20-%20KPU%20Strategic%20Plan.pdf",
    "https://www2.unbc.ca/sites/default/files/sections/strategic-planning/unbc_strategic_plan_2024-2029_final.pdf",
    "https://adm.viu.ca/sites/default/files/2021-2026-strategic-plan.pdf",
    "https://www.bcit.ca/files/financialservices/pdf/financial-statements-2023.pdf",
    "https://langara.ca/about-langara/academic-plan/documents/Langara-Academic-Plan-2023-2028.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/standards_manual.pdf",
]

# Process all documents (run once to build the database)
if not st.session_state.get("database_built", False):
    all_chunks = []
    chunk_ids = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    for url in bc_policy_urls:
        text = fetch_and_extract(url)
        if text:
            chunks = text_splitter.split_text(text)
            chunks_with_meta = [f"Source: {url.split('/')[-1]}\n\n{chunk}" for chunk in chunks]
            all_chunks.extend(chunks_with_meta)
            chunk_ids.extend([f"{url.split('/')[-1]}_{i}" for i in range(len(chunks))])
            print(f"Processed {url}: {len(chunks)} chunks")

    # Batch upsert to Pinecone
    index = pc.Index(INDEX_NAME)
    BATCH_SIZE = 100
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch_vectors = [(chunk_ids[j], embeddings.embed_query(all_chunks[j]), {"text": all_chunks[j]}) 
                         for j in range(i, min(i + BATCH_SIZE, len(all_chunks)))]
        index.upsert(vectors=batch_vectors)
        print(f"Upserted batch {i // BATCH_SIZE + 1} of {(len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE}")
    st.session_state.database_built = True
    print("Database uploaded to Pinecone index 'bc-policy-db'")

# OpenRouter LLM class
class OpenRouterLLM(LLM):
    api_key: str
    model: str = "deepseek/deepseek-chat-v3-0324:free"

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

# Initialize embeddings and LLM
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
api_key = "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"
llm = OpenRouterLLM(api_key=api_key)

# Load the Pinecone vector store
index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3})
)

# Streamlit chatbot interface
st.title("BC Policy Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I’m your BC Policy Chatbot, powered by Pinecone. Ask me anything about funding, policies, or goals across BC institutions!"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about BC policies"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing across BC institutions..."):
            response = qa_chain.run(prompt)
            st.markdown(response)
        # Add response to history
        st.session_state.messages.append({"role": "assistant", "content": response})