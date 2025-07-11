import os
from langchain_core.documents import Document
import pypdf
import docx2txt

def load_resumes(folder_path):
    documents = []
    doc_ids = []
    for filename in os.listdir(folder_path):
        try:
            file_path = os.path.join(folder_path, filename)
            if filename.endswith(".pdf"):
                reader = pypdf.PdfReader(file_path)
                text = "\n".join([page.extract_text() for page in reader.pages])
            elif filename.endswith(".docx"):
                text = docx2txt.process(file_path)
            else:
                continue
            documents.append(Document(page_content=text, metadata={"source": filename}))
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    # print(f"Loaded {len(documents)} documents from {folder_path}")
    return documents
