from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite'
)

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You're helpful and expert assistant in {domain}"),
    ("human", "Tell me about on topic {topic}")
])

prompt = chat_template.invoke({'domain': 'Doctor', 'topic': 'Fever'})
result=model.invoke(prompt)
print(result.text)