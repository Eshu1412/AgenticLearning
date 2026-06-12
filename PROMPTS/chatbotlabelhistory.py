from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

message=[
    SystemMessage(content='You\'re helpful AI assistance'),
   
]
while True:
    user_input=input("You:")
    message.append(HumanMessage(content=user_input))
    result=model.invoke(message)
    message.append(AIMessage(content=result.text))
    print(f'AI: {result.text}')
    