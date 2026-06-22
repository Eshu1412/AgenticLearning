#This code is demostrate that with_strucuture_output is working or not with HF model used in this script (Not Working)
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from pydantic import BaseModel,Field
from typing import Optional,Literal
from dotenv import load_dotenv

load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task='text-generation',
    temperature=0
)
model=ChatHuggingFace(llm=llm)
class Review(BaseModel):
    key_theme:list[str]=Field(description='Write down all the key theme disscuss on the highlight')
    summary:str=Field(description="Write down the short about 15 words summary")
    reviewer_name:str=Field("Write the Reviewer name")
    rating:int=Field(gt=1,lt=5,description='rating of the review out of 5')
    sentiment:Literal["negative","positive"]=Field(description='Return sentiment of the review either postive or negative')
    pros:Optional[list[str]]=Field(description='Write all the pros of the review if available')
    cons:Optional[list[str]]=Field(description='Write all the cons of the review if available')

structure_model=model.with_structured_output(Review)
review="""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

Reviewed by Tushar Maurya
"""
result=structure_model.invoke(review)
result=result.model_dump()
for key,value in result.items():
    print(f'{key}:{value}')