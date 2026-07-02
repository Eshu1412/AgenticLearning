from typing import Literal
from pydantic import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda

load_dotenv()
class FeedBack(BaseModel):
    sentiment: Literal['positive', 'negative']=Field(description='Give the sentiment of the user feedback only positive or negative')
pydantic_outparser=PydanticOutputParser(pydantic_object=FeedBack)
model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0
)
parser=StrOutputParser()
prompt1=PromptTemplate(
    template='You are a sentiment analyzer of user feedback. Respond with either positive or negative do not suggest anything else.\n {feedback}\n{format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':pydantic_outparser.get_format_instructions()}
)
prompt2=PromptTemplate(
    template='''
    You are a customer support representative.
    A customer left the following POSITIVE feedback:{feedback}
    Write ONLY one professional thank-you response from the company.
    Rules:
    - Return exactly one response.
    - Do NOT explain.
    - Do NOT provide multiple options.
    - Do NOT provide tips.
    - Do NOT use markdown.
    - Mentioned the positive point in response
    ''',
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template="""
    You are a customer support representative.
    A customer left the following NEGATIVE feedback:
    {feedback}
    Write ONLY one professional apology and support response from the company.
    Rules:
    - Return exactly one response.
    - Do NOT explain.
    - Do NOT provide multiple options.
    - Do NOT provide tips.
    - Do NOT use markdown.
    - Mention the negative point in response (but in polite way)
    """,
    input_variables=['feedback']
)

classifier_model=prompt1|model|pydantic_outparser
# result=classifier_model.invoke({'feedback':'The phone have most durable battery and ultra fast speed  but average display life i ever seen'}).sentiment

runnable_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2 | model | parser),
    (lambda x:x.sentiment=='negative',prompt3 | model | parser),
    RunnableLambda(lambda x:' Unknown Feedback')
)
chain = classifier_model | runnable_chain
print(chain.invoke({'feedback':'The phone have most worst battery and slow speed  and average display quality i ever seen'}))
