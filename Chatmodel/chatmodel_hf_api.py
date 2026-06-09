from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation',
    temperature=0
)
model=ChatHuggingFace(llm=llm)
response=model.invoke('Tell me about famous Lucknow Tunday Kababi')
print(response.content)