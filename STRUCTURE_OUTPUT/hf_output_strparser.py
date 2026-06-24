from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation',
    temperature=0
)
model=ChatHuggingFace(llm=llm)
#prompt1
template1=PromptTemplate(
    template='Write detailed report on {topic}',
    input_variables=['topic']
)
template2=PromptTemplate(
    template='write a 5 line summary on the following text\n {text}',
    input_variables=['text']
)
parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'AI Cyber War'})


