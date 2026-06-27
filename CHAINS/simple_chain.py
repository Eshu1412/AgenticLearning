from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')
prompt=PromptTemplate(
    template='Write 5 instersting point about topic  into short lines{topic}',
    input_variables=['topic']
)
parser=StrOutputParser()

chain=prompt|model|parser
print(chain.invoke({'topic':'Benfits of Biotechnology with AI'}))
chain.get_graph().print_ascii()



