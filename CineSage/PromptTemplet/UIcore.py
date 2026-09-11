
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

model = ChatMistralAI(model="ministral-14b-2512")

promt = ChatPromptTemplate.from_messages([
    ("system",
     """
      You are a professional movie extraction assistant.

      Your task:
      Extract the useful structured information from the movie paragraph and present in a clean redable format.

      Rules:
      - Do not add explanation
      - Do not add extra commentary
      - Follow the exact format
      - If the information is missing -> write "Not Available"
      - keep summary short (2 to 3 lines max)
      - Do not guess unknown facts
      - Maintain output format as it is in seperate lines

      Output Format:

      Movie title:
      Release year:
      Genre:
      Director:
      Main Cast:
      Setting/Location:
      Plot:
      Theme:
      Rating:
      Notable features:

      Short summary:

      """),
    ("human", "Extract the information from the following movie paragraph: {movie_paragraph}")
])


# Streamlit UI
st.title("🎬 Movie Information Extractor")

movie_paragraph = st.text_area(
    "Enter the movie paragraph:",
    height=250
)

if st.button("Extract Information"):
    if movie_paragraph:
        final_prompt = promt.invoke({
            "movie_paragraph": movie_paragraph
        })

        response = model.invoke(final_prompt)

        st.write(response.content)
    else:
        st.warning("Please enter a movie paragraph.")

