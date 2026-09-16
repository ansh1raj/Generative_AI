# -------- Token based splitting ----------

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import TokenTextSplitter

# text_splitter = TokenTextSplitter(
#     chunk_size=1000, 
#     chunk_overlap=10
#     )


# data = PyPDFLoader("RAG/document loaders/GRU.pdf")
# docs = data.load()

# # print(docs[0].page_content)
# # print(len(docs))

# chunks = text_splitter.split_documents(docs)
# print(chunks[0].page_content)


# ------------- RecursiveCharacterTextSplitter -------------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=10
    )


data = PyPDFLoader("RAG/document loaders/GRU.pdf")
docs = data.load()

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)