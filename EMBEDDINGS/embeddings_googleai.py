from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2",dimensions=32)

query_vector=embeddings.embed_query("What is Langchain?")

print(query_vector)



