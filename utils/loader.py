import os
import docx2txt
import PyPDF2
from langchain_core.documents import Document


def load_resumes(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif filename.endswith(".docx") or filename.endswith(".doc"):
            reader = docx2txt.process(file_path)
        else:
            continue

        documents.append(Document(page_content=text, metadata={"source": filename}))
    return documents
