from pydantic import Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser 
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel,Field
load_dotenv()

#llm model 
llm=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite')

##Pydantic Schema
class Person(BaseModel):
    name:str=Field(description="Write the name of the person")
    age:int=Field(gt=18,description="Write the age of the person")
    gender:str=Field(description="Write the gender of the person")
    occupation:str=Field(description='write down the profession of the person')
    city:str=Field(description="Write the city")
#Pydantic Parser
parser=PydanticOutputParser(pydantic_object=Person)

#Pydantic Prompt Template
template=PromptTemplate(
    template="Write down the information about the person of {city}\n{format_instruction}",
    input_variables=['city'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
#Prompt Template invoke
prompt=template.invoke({'city':'Florida'})

#LLM invoke
result=llm.invoke(prompt)
print(result.text)