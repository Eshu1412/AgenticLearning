from langchain_community.document_loaders import TextLoader
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()
file_path=os.path.join(os.path.dirname(__file__),'sample_doc.txt')
loader=TextLoader(file_path,encoding='utf-8')
docs=loader.load()
# print(docs[0].page_content)

model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0
)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template='write a summary of 100-120 words of the given  {text}',
    input_variables=['text']
)
chain=RunnableSequence(prompt1,model,parser)
result=chain.invoke({'text':docs[0].page_content})
print(result)

