from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    dimensions=63
)

texts = [
    "hello i am ansh raj",
    "who are you",
    "and are ok?"
]
# vector = embeddings.embed_query("Hello world")

vector = embeddings.embed_documents(texts)
print(vector)
