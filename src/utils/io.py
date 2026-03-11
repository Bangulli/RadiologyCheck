import pdfplumber
#gastrofrom docx import Document

def extract_text(filepath):
    if filepath.endswith(".pdf"):
        with pdfplumber.open(filepath) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages)
    # elif filepath.endswith(".docx"):
    #     doc = Document(filepath)
    #     return "\n".join(para.text for para in doc.paragraphs)
    else:  # plain .txt
        with open(filepath, "r") as f:
            return f.read()