from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables  import (
RunnableBranch,
RunnableSequence,
RunnablePassthrough)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()
prompt1=PromptTemplate(
    template='Write a report on the {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Write a short summary of the following text:\n\n{text}',
    input_variables=['text']
)
model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0
)
parser=StrOutputParser()
report=RunnableSequence(prompt1,model,parser)
chain_branch=RunnableBranch(
    (lambda x: len(x.split())>50 , (lambda x: {'text': x}) | RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)
result=RunnableSequence(report,chain_branch)
output = result.invoke({'topic':'AI in Civil Engineering'})
print(output)