from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = "ministral-3b-2512",temperature=0.7)

print("welcome to mode based chatbot")
print("press the mode of chatbot")
print("press 1 for angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")

choice = int(input("Enter your choice: "))

if choice == 1:
   mode = "you are an angry AI agent. you are very rude and sarcastic. you will answer in a very angry way in short words"
elif choice == 2:
   mode = "you are a funny AI agent. you will answer in a very funny way in short words"
elif choice == 3: 
   mode = "you are a sad AI agent. you will answer in a very sad way in short words"

messages = [
    SystemMessage(content= mode)
]

print("-------Enter '0' to exit the chat--------")
while True:
 prompt = input("You: ")
 messages.append(HumanMessage(content= prompt))
 if(prompt == "0"):
  break
 response = model.invoke(messages)
 messages.append(AIMessage(content= response.content))
 print("Bot: ",response.content)