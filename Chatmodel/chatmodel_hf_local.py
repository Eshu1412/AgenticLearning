from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
import os
from dotenv import load_dotenv


load_dotenv()

os.environ['HF_HOME']='C:\\huggingface_cache'
llm=HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100,
    )
    
)
model=ChatHuggingFace(llm=llm)

result=model.invoke("what is fastapi tell me under 100 tokens")
print(result.content)