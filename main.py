from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma


model = ChatMistralAI(model = "ministral-14b-2512")


template = ChatPromptTemplate.from_messages([
    ("system","you are an AI that summarizes the text"),
    ("human","{human_data}")
])





