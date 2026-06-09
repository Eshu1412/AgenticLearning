from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm=GoogleGenerativeAI(model='gemini-2.5-flash')
result=llm.invoke('Tell me about Tundey Kabibi? In Lucknow')
print(result)
# llm=OpenAI(model='gpt-3.5-turbo-instruct')
# llm.invoke("What is the capital of India")