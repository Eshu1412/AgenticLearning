from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough,RunnableSequence,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()

prompt1=PromptTemplate(
    template='Write joke on topic (RESTRICT markdown and speical symbol like #,* etc) {topic}',
    input_variables=['topic'])

prompt2=PromptTemplate(
    template='Explain the joke in details (RESTRICT markdown and speical symbol like #,* etc word limit 60-120){Jokes}',
    input_variables=['Jokes'])

markdown_prompt=PromptTemplate(
    template='''Convert the following text to mardown text make the key (dict) into heading
     and do not add any explanation just convert it into markdown all the heading must be in UPPERCASE {text}''',
    input_variables=['text']
)

parser=StrOutputParser()
model=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite'
)
joke_sequence=RunnableSequence(prompt1,model,parser)

parallel_runnable=RunnableParallel({
    'jokes':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2,model,parser)
})
final_chain=RunnableSequence(joke_sequence,parallel_runnable)
result=final_chain.invoke({'topic':'AI in the old stone age (Paleolithic Period)'})
print(result['jokes'])
print('-'*50)
print(result['explanation'])
markdown_text=RunnableSequence(markdown_prompt,model,parser)
mardown_result=markdown_text.invoke({'text':result})

current_path=os.path.dirname(os.path.abspath(__file__))
markdown_file_path=os.path.join(current_path,'Joke.md')
with open(markdown_file_path,'w',encoding='utf-8') as f:
    f.write(mardown_result)