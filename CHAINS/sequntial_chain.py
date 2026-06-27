from markdown_it import parser_core
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite'
)
parser=StrOutputParser()
prompt1=PromptTemplate(
    template='Generate a detailed report on topic {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Generate a crucial 5 points from the given text {text} 10-20 words.',
    input_variables=['text']
)
chain=prompt1 | model | parser | prompt2 | model | parser

result=chain.invoke({'Unemployment in India'})
print(result)