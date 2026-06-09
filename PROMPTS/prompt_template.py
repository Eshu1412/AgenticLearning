from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
from pathlib import Path

TEMPLATE_PATH=Path(__file__).parent/'TEMPLATE'
#load the environment variable 
load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0
)
st.header('Research Tool')
paper_input=st.selectbox("Select Research Paper Name",
                         ["Select...","Attention is all you need ",
                          "BERT:  Pre-training of Deep Biderectional Transformers",
                          "GPT-3: Language Models are few shot learners",
                          "Diffusion Models Beats GANs on Image Synthesis"])
style_input=st.selectbox("Select Explanation Style",["Beginner-Friendly","Technical Code Oreinted","Mathmatical"])
length_input=st.selectbox("Select Explanation Length",['Short (1-2 paragraphs)','Medium (3-5 paragraphs)','Long (detailed explanation)'])
#PromptTemplate in langchain
template=load_prompt(TEMPLATE_PATH/'template.json')
prompt=template.invoke({
    "paper_input":paper_input,
    "style_input":style_input,
    "length_input":length_input
})
if st.button("Summarize"):
    result=model.invoke(prompt)
    st.write(result.text)