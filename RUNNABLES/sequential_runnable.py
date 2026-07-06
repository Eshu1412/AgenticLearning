from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0.7
)
prompt1=PromptTemplate(
    template='Write a insane joke about {topic}\n IMPORTANT INSTRUCTION Joke should be in single line only No markdown symbol',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Translate the following joke into Hinglish:\n{text}',
    input_variables=['text']
)
parser=StrOutputParser()

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
result=chain.invoke({'topic':'AI vs Humans'})
print(result)
chain.get_graph().print_ascii()
