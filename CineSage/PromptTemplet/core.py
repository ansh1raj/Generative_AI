from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model = "ministral-14b-2512")

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

para = input("Enter the movie paragraph: ")
final_prompt = promt.invoke({"movie_paragraph": para})

response = model.invoke(final_prompt)
print(response.content)

# *3 Idiots* is a 2009 Indian comedy-drama film directed by Rajkumar Hirani, starring Aamir Khan, R. Madhavan, Sharman Joshi, Kareena Kapoor Khan, Boman Irani, and Omi Vaidya. The movie is mainly set at the Imperial College of Engineering in India, with other scenes taking place in locations such as Delhi and Shimla. The story follows three engineering students, Rancho, Farhan, and Raju, who become close friends while dealing with academic pressure, family expectations, and a highly competitive education system. Rancho encourages his friends to focus on learning and following their passions rather than simply chasing grades and successful careers. The main themes of the film are friendship, following one's passion, dealing with academic pressure, creativity, and questioning traditional ideas of success. *3 Idiots* has an IMDb rating of around 8.4/10 and is especially notable for its memorable comedy, emotional storytelling, strong friendship, social commentary on the Indian education system, and popular songs such as “All Is Well” and “Give Me Some Sunshine.” Overall, the film tells an inspiring story about three friends whose college experiences teach them the importance of learning for knowledge, pursuing what they love, supporting one another, and finding success on their own terms.
