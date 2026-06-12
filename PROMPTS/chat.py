from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite'
)
chat_history=[]
while True:
    user_input=input("Enter the Chat:")
    if user_input.lower()=='exit':
        break
    else:
        chat_history.append(user_input)
        response=model.invoke(chat_history)
        chat_history.append(response.text)
        print(response.text)