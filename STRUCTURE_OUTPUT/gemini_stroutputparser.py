from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite'
)
parser=StrOutputParser()

#prompt1
template1=PromptTemplate(
    template='Write a detailed report on the topic {topic}',
    input_variables=['topic']
)
#prompt2
template2=PromptTemplate(
    template='write a 5 line summary on the following text\n With following heading format: {topic}: {text}',
    input_variables=['text']
)
#chaining the prompts
chain=template1 | model | parser | template2 | model | parser

#invoke the chain
result=chain.invoke({'topic':'AI Cyber War'})
print(result)