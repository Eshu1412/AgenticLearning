from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

chatmodel=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

result=chatmodel.invoke("Tell me about Tundey Kabibi",temperature=0)
print(result)
