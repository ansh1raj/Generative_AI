from langchain_community.document_loaders import WebBaseLoader

url = "https://cluely.com/"

data = WebBaseLoader(url)
docs = data.load()

print(docs)