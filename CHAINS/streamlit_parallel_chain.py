from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
parser=StrOutputParser()
st.header("Simple parallel chain using streamlit for Notes and question generation")
input_text=st.text_input('Enter your topic')
model1=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    temperature=0
)
model2=ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0.2
)
prompt1=PromptTemplate(
    template='You are an expert in writing short notes on the following topics:\n{text}\n write only 250 words',
    input_variables=['text']
)
prompt2=PromptTemplate(
    template="You are an expert in writing quiz questions based on the text {text}. Provide 5 quiz questions in MCQ format and each question use ## heading in markdown. IMPORTANT: Each option (a, b, c, d) must be separated by a blank line (double newline) so that they render on separate lines in Markdown. Example:\n\n1. Question?\n\na) Option A\n\nb) Option B\n\nc) Option C\n\nd) Option D",
    input_variables=['text']
)
prompt3=PromptTemplate(
    template="Merge the following notes and quiz together.\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}\n\nIMPORTANT: In the final merged output, you MUST format the quiz questions so that each option (a, b, c, d) is separated by a blank line (double newline) to ensure they are displayed on separate lines in Markdown. Do NOT combine options onto a single line. Example:\n\n1. Question?\n\na) Option A\n\nb) Option B\n\nc) Option C\n\nd) Option D",
    input_variables=['notes','quiz']
)
parallel_merge=RunnableParallel(
    {
        'notes':prompt1 | model1 | parser,
        'quiz':prompt2 | model2 | parser
    }
)
merge_chain=prompt3 | model2 | parser
chain=parallel_merge | merge_chain
if st.button('Generate'):
    if input_text:
        with st.spinner('Generating Notes and Quiz...'):
            result=chain.invoke({'text':input_text})
            st.success('Notes and Quiz Generated Successfully!')
            st.write(result)
            st.subheader('Graph')
            st.code(chain.get_graph().draw_ascii())
    else:
        st.error('Please enter a topic')


        