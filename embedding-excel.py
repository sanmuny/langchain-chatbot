import os
import getpass
import glob
from dotenv import load_dotenv
from tqdm import tqdm
import pickle
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

load_dotenv(verbose=True)
document_dir = os.environ["DOCUMENT_DIR"]
vectorstore_dir = os.environ["VECTOR_DB_DIR"]
embeddings_model = os.environ["MODEL_EMBEDDINGS"]

### WORKAROUND for "trust_remote_code=True is required" error in HuggingFaceEmbeddings()
from transformers import AutoModel

model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True)


##
## Read source documents and extract return the list contains documents
##
def generate_vectorstore_docs_from_data_source(
    glob_pattern: str, max_doc_count: int = -1
):
    doc_count = 0
    data_files = glob.glob(glob_pattern, recursive=True)
    documents = []

    print("*** Begin to split documents")
    for data_file in tqdm(data_files):
        df = pd.read_excel(data_file)

        for index, row in df.iterrows():
            row_dict = row.to_dict()
            row_text = ", ".join([f"{key}: {value}" for key, value in row_dict.items()])
            document = Document(page_content=row_text)
            documents.append(document)

    print("*** Complete to split documents")

    hf_bge_embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=vectorstore_dir, embedding_function=hf_bge_embeddings
    )
    print("*** Begin to add documents into vectorDB")
    for doc in tqdm(documents):
        vectorstore.add_documents([doc])
    print("*** Complete to add documents into vectorDB")


# Generate documents from data source. Read the documents from pickle file if exists.
pickle_file = "./doc_obj.pickle"
if not os.path.exists(pickle_file):
    print(
        "*** Reading Original Data and generating document(s) and Converting documents into embeddings and creating a vector store(s)"
    )
    docs = generate_vectorstore_docs_from_data_source(f"{document_dir}/**/*.xlsx")
    with open(pickle_file, "wb") as f:
        pickle.dump(docs, file=f)
else:
    print(
        f"*** Reading documents and Converting documents into embeddings and creating a vector store(s) from a pickled file ({pickle_file})"
    )
    with open(pickle_file, "rb") as f:
        docs = pickle.load(file=f)
