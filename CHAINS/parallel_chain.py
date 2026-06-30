import grpc # Fix Windows + Python 3.14 DLL import issue in ThreadPoolExecutor
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_openrouter import ChatOpenRouter
load_dotenv()

model1=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0.5
)
model2=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=1
)

prompt1=PromptTemplate(
    template='Write short notes on following text:{text}',
    input_variables=['text']
)
prompt2=PromptTemplate(
    template='Generate simple 5 quiz question from the following text:{text}',
    input_variables=['text']
)
prompt3=PromptTemplate(
    template='Merge the following notes and quiz together notes:{notes} and quiz:{quiz}'
)
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1 | model1|parser,
    'quiz':prompt2 | model2 | parser}
)
merge_chain=prompt3 | model2| parser
chain=parallel_chain | merge_chain
result=chain.invoke({'text':'Quantum Computing'})
file_name=input("Enter the File name to save the result (e.g notes.md)\n:")
with open(file_name,'w') as f:
    f.write(result)
