import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# import textract


def load_resumes(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        try:
            file_path = os.path.join(folder_path, filename)
            loader = None
            text = None
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            # elif filename.endswith(".doc"):
            #     text = textract.process(file_path, extension='doc').decode('utf-8')
            #     documents.append(Document(page_content=text, metadata={"source": filename}))
            else:
                continue
            if loader is not None:
                docs = loader.load()
                documents.extend(docs)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        # This line is now handled above for .doc files, so it can be removed.
        # documents.append(Document(page_content=text, metadata={"source": filename}))
    print(f"Loaded {len(documents)} documents from {folder_path}")
    if not documents:
        print("No valid documents found in the specified folder.")
    else:
        print(f"Successfully loaded {len(documents)} documents.")
    return documents
