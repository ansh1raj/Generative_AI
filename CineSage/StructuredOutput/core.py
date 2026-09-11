from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = "ministral-14b-2512")

class Movie(BaseModel):
    title: str 
    release_year : Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)


promt = ChatPromptTemplate.from_messages([
 ("system","""
  Extract movie information from the paragraph
   {format_ins}
"""),
("human", "{movie_paragraph}")
])

para = input("Enter the movie paragraph: ")
final_prompt = promt.invoke({
    "movie_paragraph": para,
    "format_ins": parser.get_format_instructions()
    })

response = model.invoke(final_prompt)
movie_data = parser.parse(response.content)

print(movie_data)