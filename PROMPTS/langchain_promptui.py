from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=2
)
st.header('Langchain Prompt UI')

user_input=st.text_input("Ask me anything")

if st.button("Send"):
    result=model.invoke(user_input)
    content = result.content
    if isinstance(content, list):
        content="".join(part.get("text","") for part in content if isinstance(part,dict))
    st.write(content)