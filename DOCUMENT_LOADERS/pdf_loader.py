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
prompt1 = PromptTemplate(
    template='''
You are a strict technical resume screener for backend/full-stack developer roles at a competitive product company. You are NOT here to encourage the candidate. You are here to filter out weak resumes before they waste a recruiter's time.

Resume content:
{content}

Scoring rubric (be harsh — most resumes should NOT score above 70):
- Relevant technical depth (0-30): Real projects with measurable impact vs buzzword listing
- Skill-role fit (0-20): Do skills match a coherent role, or is it a scattershot of unrelated tags
- Evidence quality (0-20): Quantified outcomes, deployed links, GitHub proof vs vague claims
- Resume clarity/structure (0-15): Can a recruiter parse this in 10 seconds
- Red flags (0-15, subtract for): buzzword stuffing, unverifiable claims, inconsistent tech stack, no live/deployed proof

Rules:
- Do NOT give a score above 85 unless the resume has verifiable, deployed, quantified work.
- If skills are listed but no project demonstrates them, treat that skill as decorative, not credited.
- If you notice inflated claims (e.g. "increased efficiency by 40%" with no context), call it out explicitly in a "Concerns" field.
- Do not soften language. If the resume is mediocre, say mediocre and say why.

Output strictly in this format:
Name: (Candidate Name)
Preferred Role: (Best Fit Role — say "Unclear" if skills don't cohere into one role)
College Name: (Name of college)
Resume Score: (0-100, per rubric above)
Score Breakdown: (one line per rubric category with points awarded)
Frontend Skills: [skill1, skill2, ...]
Backend Skills: [skill1, skill2, ...]
Database: [skill1, skill2, ...]
Concerns: (list any unverifiable claims, buzzword stuffing, or skill-role mismatch — say "None" only if genuinely none)
''',
    input_variables=['content']
)
chain=RunnableSequence(prompt1,model,StrOutputParser())
result=chain.invoke({'content':full_content})
print(result)
