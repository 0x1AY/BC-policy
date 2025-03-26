import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import streamlit as st
from typing import Optional, List, Mapping, Any

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
    "https://universitycounsel.ubc.ca/files/2022/05/Research-Policy_LR2.pdf",  # UBC Research Policy
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
    'https://www.capilanou.ca/media/capilanouca/about-capu/governance/presidentx27s-office/reports-amp-initiatives/envisioning-2030/Envisioning-2030-Report.pdf',
    'https://www.capilanou.ca/media/capilanouca/about-capu/governance/budget-plans-amp-reports/financial-reports/Audited-Financial-Statements-2023-24.pdf',
    'https://www.tru.ca/__shared/assets/science_strategic_plan_2021-202654932.pdf',
    'https://www.tru.ca/__shared/assets/2023-2024_Budget57554.pdf',
    'https://www.tru.ca/__shared/assets/BRD_10-0_Academic_Accommodations__No_Date_61891.pdf',
    'https://www.tru.ca/__shared/assets/ED_09-1_Academic_Achievement_Awards__2023-06-12_61892.pdf',
    'https://www.tru.ca/__shared/assets/ADM_04-5_Advertising__2007-09-04_5584.pdf',
    'https://www.tru.ca/__shared/assets/ADM_05-3_Alcohol__Cannabis__and_Tobacco__2018-09-26_5601.pdf',
    'https://www.tru.ca/__shared/assets/Policy_ADM_25-0_Biosafety_and_Biosecurity__2017-02-28_40206.pdf',
    'https://www.kpu.ca/sites/default/files/Institutional%20Analysis%20and%20Planning/VISION%202026%20-%20DRAFT%20Jan%2030%202023.pdf',
    'https://www.kpu.ca/sites/default/files/Policies/HR14%20Employment%20Students%20Policy.pdf',
    'https://www.kpu.ca/sites/default/files/Policies/HR16%20Employment%20Equity%20Policy.pdf',
    'https://www.kpu.ca/sites/default/files/Policies/IM1%20Copyright%20Compliance%20Policy.pdf',
    'https://www.kpu.ca/sites/default/files/Policies/RS1%20Research%20Involving%20Human%20Participants%20Policy.pdf',
    'https://www.kpu.ca/sites/default/files/Policies/SR9%20Violence%20in%20the%20Workplace.pdf',
    'https://www.unbc.ca/sites/default/files/sections/strategic-planning/ready_fulldoc.pdf',
    'https://www.unbc.ca/sites/default/files/sections/finance/budgets/unbcbudget2024-25bogfinal.pdf',
    'https://www.unbc.ca/sites/default/files/sections/finance/budgets/unbc-budget-2025-26-bog-final_0.pdf',
    'https://gov.viu.ca/sites/default/files/viu-strategic-plan.pdf',
    'https://gov.viu.ca/sites/default/files/fy2024-25-operating-budget-report.pdf',
    'https://isapp.viu.ca/policyprocedure/docshow.asp?doc_id=21090',
    'https://isapp.viu.ca/policyprocedure/docshow.asp?doc_id=21091',
    'https://www.bcit.ca/files/pdf/policies/1000.pdf',
    'https://www.bcit.ca/files/pdf/policies/6505_animal_care_policy_v1.pdf',
    'https://www.bcit.ca/files/pdf/policies/6505-pr1_procedures_for_ethical_use_of_animals_v1.pdf',
    'https://www.bcit.ca/files/pdf/policies/7200.pdf',
    'https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/standards_manual.pdf',
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/bcit_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/cam_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/capu_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/cmtn_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/cnc_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/cotr_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/doug_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/ecu_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/jibc_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/kpu_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/lang_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/nvit_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/nic_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/nlc_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/okan_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/rru_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/sel_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/sfu_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/tru_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/ubc_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/unbc_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/ufv_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/uvic_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/vcc_iapr.pdf",
    "https://www2.gov.bc.ca/assets/gov/education/post-secondary-education/institution-resources-administration/accountability-framework/iapr/viu_iapr.pdf"

]

# Process all documents (run this section once to build the database)
if not st.session_state.get("database_built", False):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    for url in bc_policy_urls:
        text = fetch_and_extract(url)
        if text:
            chunks = text_splitter.split_text(text)
            chunks_with_meta = [f"Source: {url.split('/')[-1]}\n\n{chunk}" for chunk in chunks]
            all_chunks.extend(chunks_with_meta)
            # print(f"Processed {url}: {len(chunks)} chunks")

    vector_store = FAISS.from_texts(all_chunks, embeddings)
    vector_store.save_local("bc_policy_db")
    st.session_state.database_built = True
    print("Database saved as 'bc_policy_db'")

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

# Load the pre-built database
vector_store = FAISS.load_local("bc_policy_db", embeddings, allow_dangerous_deserialization=True)
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
        {"role": "assistant", "content": "Hi! I’m your BC Policy Chatbot. Ask me anything about funding, policies, or goals across BC institutions!"}
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