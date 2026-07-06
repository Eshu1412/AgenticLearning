from langchain_core.prompts import PromptTemplate
from torch.cuda import temperature
from langchain_core.runnables import RunnableParallel,RunnableSequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

gemini_model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
)
hf_llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation',
    temperature=0.3
)
hf_model=ChatHuggingFace(llm=hf_llm)
parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Create a tweet post 10-30 words on topic do not include any special symbol or emoji: {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Create a facebook 10-30 words on topic do not include any special symbol or emoji provide revenlant professional response restrict suggestion and option: {topic}',
    input_variables={'topic'}
)

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt1,gemini_model,parser),
    'facebook':RunnableSequence(prompt2,hf_model,parser)
    })
result=parallel_chain.invoke({'topic':'AI in Cybersecurity'})
for i,j in result.items():
    print(f'{i}:{j}')
    print()
