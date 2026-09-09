from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

texts = [
    "hello i am ansh raj",
    "who are you",
    "and are ok?"
]

vector = embeddings.embed_documents(texts)
print(vector)
