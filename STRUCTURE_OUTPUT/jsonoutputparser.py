from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation',
    temperature=1.5
)
chat_model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()
template1=PromptTemplate(
    template="Give me the name,profession,city and age of fictional character \n{format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

result=chat_model.invoke(template1)
final_result=parser.parse(result.content)
print(final_result)