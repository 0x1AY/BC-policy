import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os

# Step 1: Get college list
url = "https://www2.gov.bc.ca/gov/content/education-training/post-secondary-education/find-a-program-institution/public-institutions"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
colleges_heading = soup.find("h2", text="Colleges")
colleges_list = colleges_heading.find_next_sibling("ul", class_="list--unstyled")
colleges = [{"name": li.find("a").text.strip(), "url": li.find("a")["href"]} for li in colleges_list.find_all("li") if li.find("a")]

# Keywords
about_keywords = ["about", "governance", "administration"]
reports_keywords = ["reports", "publications", "finance"]
academics_keywords = ["academics", "programs", "courses"]
strategic_keywords = ["strategic plan", "institutional plan"]
financial_keywords = ["financial report", "annual report"]
curriculum_keywords = ["academic calendar", "course catalog"]

# Functions
def find_section_links(base_url, section_keywords):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, "html.parser")
    return [requests.compat.urljoin(base_url, a["href"]) for a in soup.find_all("a", href=True) if any(k in a.text.lower() for k in section_keywords)]

def find_pdfs_in_page(page_url, keywords):
    try:
        response = requests.get(page_url)
        soup = BeautifulSoup(response.content, "html.parser")
        return [requests.compat.urljoin(page_url, a["href"]) for a in soup.find_all("a", href=True) if any(k in a["href"].lower() for k in keywords) and a["href"].lower().endswith(".pdf")]
    except:
        return []

def download_pdf(pdf_url, save_path):
    try:
        response = requests.get(pdf_url)
        with open(save_path, "wb") as file:
            file.write(response.content)
    except:
        print(f"Failed to download {pdf_url}")

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

# Process each college
for college in colleges[:5]:  # Limit to 5 for demo
    print(f"\nProcessing {college['name']} ({college['url']})")
    save_dir = f"data/{college['name'].replace(' ', '_')}"
    os.makedirs(save_dir, exist_ok=True)

    # Find sections
    about_sections = find_section_links(college["url"], about_keywords)
    reports_sections = find_section_links(college["url"], reports_keywords)
    academics_sections = find_section_links(college["url"], academics_keywords)

    # Find and download PDFs
    for section, keywords, prefix in [
        (about_sections, strategic_keywords, "strategic"),
        (reports_sections, financial_keywords, "financial"),
        (academics_sections, curriculum_keywords, "curriculum")
    ]:
        for page in section:
            pdfs = find_pdfs_in_page(page, keywords)
            for pdf_url in pdfs:
                filename = f"{prefix}_{pdf_url.split('/')[-1]}"
                save_path = os.path.join(save_dir, filename)
                download_pdf(pdf_url, save_path)
                text = extract_text_from_pdf(save_path)
                print(f"Extracted {len(text)} chars from {filename}")