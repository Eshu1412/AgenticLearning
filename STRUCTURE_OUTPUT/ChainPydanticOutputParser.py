from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite',temperature=0.7)

class Student(BaseModel):
    name:str=Field(description='Name of the student')
    age:int=Field(description='Age of the student in integer')
    grade:str=Field(description='Grade of the student in string (A-F)')
    city:str=Field(description='City name of the student')
    course:str=Field(description='Course selected by the student')
    branch:str=Field(description='Branch of the student')

student_parser=PydanticOutputParser(pydantic_object=Student)
student_prompt_template=PromptTemplate(
    template='Write down the information of the student basis of their city name:{city}\n{format_instructions}',
    input_variables=['city'],
    partial_variables={'format_instructions':student_parser.get_format_instructions()}
)
chain=student_prompt_template|model|student_parser
result=chain.invoke({'city':'Los Santos'}).model_dump() #model_dump for dictionary
print(result)
