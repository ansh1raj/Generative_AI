from dotenv import load_dotenv

load_dotenv()

# ----init_chat_model--------

# from langchain.chat_models import init_chat_model
# model = init_chat_model(
#     "groq:openai/gpt-oss-120b"
# )


# ------- Model Class ---------

# from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatGoogleGenerativeAI(model="gemini-3.7-flash")

from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = "ministral-3b-2512",temperature=0.7, max_tokens=20)

response = model.invoke("give me a short note on machine learning")
print(response.content)
