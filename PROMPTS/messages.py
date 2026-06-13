from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',temperature=0.5)
chat_template=ChatPromptTemplate(
    [('system','You\'re helpful customer support assistance'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
    ]
)
chat_history=[]
with open('PROMPTS/message_data.txt','r') as f:
    chat_history.extend(f.readlines())

chat=chat_template.invoke({'chat_history':chat_history,'query':'where is my refund'})
response=model.invoke(chat)
print(response.text)


