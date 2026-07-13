from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

file_path=os.path.join(os.path.dirname(__file__),'Tushar_Maurya_CV.pdf')
loader=PyPDFLoader(file_path)
docs=loader.load()
full_content='\n'.join([doc.page_content for doc in docs])
model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0
)
prompt1=PromptTemplate(
    template='''
    Create a candidate details and perfect suitable role on the basis of following content\n
     {content} 
     Note the following format should be followed
     Name:(Candidate Name)
     Preferred Role: (Best Fit Role)
     College Name: (Name of college)
     Resume Score: (score of resume out of 100)
     Frontend Skills: [skill1,skill2,skill3...]
     Backend Skills:[skill1,skill2,skill3...]
     Database:[skill1,skill2,skill3...]
     ''',
    input_variables=['content']
)
chain=RunnableSequence(prompt1,model,StrOutputParser())
result=chain.invoke({'content':full_content})
print(result)
