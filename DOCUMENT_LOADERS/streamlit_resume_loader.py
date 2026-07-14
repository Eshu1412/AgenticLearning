import streamlit as st
import tempfile
import os
import re
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────── CSS ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; font-family: 'Inter', sans-serif; }

/* Background */
.stApp { background: #0d0d0d; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.block-container { padding: 2.5rem 2.5rem 4rem; max-width: 960px; }

/* ── Hero ── */
.hero { padding: 2rem 0 1.5rem; border-bottom: 1px solid #1f1f1f; margin-bottom: 2rem; }
.hero-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #f1f1f1;
    margin: 0 0 0.35rem;
    letter-spacing: -0.5px;
}
.hero p { color: #555; font-size: 0.9rem; margin: 0; }

/* ── Upload area ── */
.upload-indicator {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #141414;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    margin: 1rem 0;
}
.upload-indicator .fname { color: #e5e5e5; font-size: 0.9rem; font-weight: 500; }
.upload-indicator .fmeta { color: #444; font-size: 0.78rem; }

/* ── Section heading ── */
.sec-heading {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #444;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1a1a;
}

/* ── Score block ── */
.score-wrap {
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 2rem 1.5rem;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
}
.score-num { font-size: 4rem; font-weight: 700; line-height: 1; letter-spacing: -2px; }
.score-sub { font-size: 0.7rem; color: #444; letter-spacing: 2px; text-transform: uppercase; margin-top: 0.4rem; }

/* ── Info card ── */
.info-card {
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
}
.info-card .label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 0.3rem;
}
.info-card .value { color: #e5e5e5; font-size: 0.95rem; font-weight: 500; }

/* ── Breakdown ── */
.breakdown {
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
}
.breakdown .label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 0.8rem;
}
.bd-row {
    font-size: 0.85rem;
    color: #888;
    padding: 0.35rem 0;
    border-bottom: 1px solid #1a1a1a;
    line-height: 1.4;
}
.bd-row:last-child { border-bottom: none; }

/* ── Skills card ── */
.skills-card {
    background: #111;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.skills-card .label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #444;
    margin-bottom: 0.75rem;
}
.chip-wrap { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    color: #aaa;
    padding: 0.2rem 0.7rem;
    border-radius: 4px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Concerns ── */
.concerns {
    background: #120d0d;
    border: 1px solid #2c1515;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
}
.concerns .label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7f1d1d;
    margin-bottom: 0.5rem;
}
.concerns p { color: #888; font-size: 0.87rem; line-height: 1.7; margin: 0; }

/* ── Button ── */
.stButton > button {
    background: #6366f1 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.5rem !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #4f46e5 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Role → skill sections ───────────────────
ROLE_SKILL_SECTIONS = {
    "Backend Developer":          ["Backend Skills",       "DevOps & Infra",       "Database"],
    "Frontend Developer":         ["Frontend Skills",      "UI Frameworks",        "Build & Dev Tools"],
    "Full Stack Developer":       ["Frontend Skills",      "Backend Skills",       "Database"],
    "Data Analyst":               ["Languages & Libraries","BI & Visualisation",   "Database"],
    "Data Scientist":             ["ML & AI Libraries",    "Languages",            "Data & Cloud Tools"],
    "Machine Learning Engineer":  ["ML & AI Libraries",    "Languages",            "MLOps & Infra"],
    "Cloud Engineer":             ["Cloud Platforms",      "DevOps Tools",         "Languages & Scripting"],
    "DevOps Engineer":            ["DevOps Tools",         "Cloud Platforms",      "Languages & Scripting"],
    "AI / LLM Engineer":          ["LLM & AI Frameworks",  "Backend Skills",       "MLOps & Tools"],
    "Mobile Developer":           ["Mobile Frameworks",    "Languages",            "Backend & APIs"],
    "QA / Test Engineer":         ["Testing Frameworks",   "Languages",            "Tools & Platforms"],
    "Product Manager":            ["Product Tools",        "Analytics Tools",      "Communication Tools"],
}

PROMPT_TEMPLATE = '''
You are a strict technical resume screener evaluating a candidate for a {role} role at a competitive product company. You are NOT here to encourage the candidate. You are here to filter out weak resumes before they waste a recruiter's time.

Resume content:
{content}

Scoring rubric specifically for a {role} role (be harsh — most resumes should NOT score above 70):
- Relevant technical depth (0-30): Real projects with measurable impact vs buzzword listing
- Skill-role fit (0-20): Do skills actually match the {role} role, or is it a scattershot of unrelated tags
- Evidence quality (0-20): Quantified outcomes, deployed links, GitHub proof vs vague claims
- Resume clarity/structure (0-15): Can a recruiter parse this in 10 seconds
- Red flags (0-15, subtract for): buzzword stuffing, unverifiable claims, inconsistent tech stack, no live/deployed proof

Rules:
- Do NOT give a score above 85 unless the resume has verifiable, deployed, quantified work.
- If skills are listed but no project demonstrates them, treat that skill as decorative, not credited.
- If you notice inflated claims (e.g. "increased efficiency by 40%" with no context), call it out explicitly in a "Concerns" field.
- Do not soften language. If the resume is mediocre, say mediocre and say why.
- Judge strictly against the {role} role — penalise skills that are irrelevant to it.
- Categorise skills ONLY into the three sections listed below — do NOT use generic labels like "Backend Skills" or "Frontend Skills" unless they appear in the section list.

Output strictly in this format:
Name: (Candidate Name)
Preferred Role: (Best Fit Role — say "Unclear" if skills don\'t cohere into one role)
College Name: (Name of college)
Resume Score: (0-100, per rubric above)
Score Breakdown: (one line per rubric category with points awarded)
{section1}: [skill1, skill2, ...]
{section2}: [skill1, skill2, ...]
{section3}: [skill1, skill2, ...]
Concerns: (list any unverifiable claims, buzzword stuffing, or skill-role mismatch — say "None" only if genuinely none)
'''

# ─────────────────────────── Helper: parse result ────────────────────
def extract_field(text: str, field: str) -> str:
    pattern = rf"^{re.escape(field)}:\s*(.+)"
    match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match else "—"

def extract_skills(text: str, field: str) -> list[str]:
    raw = extract_field(text, field)
    raw = raw.strip("[]")
    return [s.strip() for s in raw.split(",") if s.strip() and s.strip() != "—"]

def extract_breakdown(text: str) -> list[str]:
    match = re.search(r"Score Breakdown:\s*(.*?)(?=\nFrontend Skills:|\nBackend Skills:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    lines = [l.strip().lstrip("-").strip() for l in match.group(1).strip().splitlines() if l.strip()]
    return lines

def extract_concerns(text: str) -> str:
    match = re.search(r"Concerns:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "—"

def md_to_html(text: str) -> str:
    """Convert basic markdown (bold, bullets) to HTML for safe inline rendering."""
    # **bold** → <strong>bold</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong style="color:#e5e5e5">\1</strong>', text)
    # leading dash bullets → styled list items
    lines = text.split("\n")
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            html_lines.append(f'<div style="padding:0.25rem 0 0.25rem 0.5rem; border-left:2px solid #2c1515; margin:0.2rem 0">{stripped[2:]}</div>')
        elif stripped:
            html_lines.append(f'<span>{stripped}</span><br>')
    return "".join(html_lines)

def score_color(score: int) -> str:
    if score >= 80: return "#34d399"
    if score >= 60: return "#60a5fa"
    if score >= 40: return "#fbbf24"
    return "#f87171"

# ─────────────────────────── LangChain setup ─────────────────────────
@st.cache_resource
def get_chain():
    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.1)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["content", "role", "section1", "section2", "section3"]
    )
    return RunnableSequence(prompt, model, StrOutputParser())

def invoke_with_retry(chain, payload: dict, retries: int = 3, delay: float = 2.0) -> str:
    """Retries chain.invoke up to `retries` times on empty/error responses."""
    for attempt in range(1, retries + 1):
        try:
            result = chain.invoke(payload)
            if result and result.strip():
                return result
            st.warning(f"Attempt {attempt}: empty response received, retrying...")
        except Exception as e:
            if attempt == retries:
                raise
            st.warning(f"Attempt {attempt} failed ({e}), retrying...")
        time.sleep(delay)
    raise RuntimeError("Model returned empty output after all retries. Try again.")

MAX_CONTENT_CHARS = 12_000  # prevent overwhelming the model with huge PDFs

def load_pdf(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    os.unlink(tmp_path)
    content = "\n".join([doc.page_content for doc in docs])
    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS] + "\n...[truncated]"
    return content

# ─────────────────────────── UI ──────────────────────────────────────
# Role options
ROLES = [
    "Backend Developer",
    "Frontend Developer",
    "Full Stack Developer",
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "Cloud Engineer",
    "DevOps Engineer",
    "AI / LLM Engineer",
    "Mobile Developer",
    "QA / Test Engineer",
    "Product Manager",
]

st.markdown("""
<div class="hero">
    <div class="hero-label">Gemini · LangChain</div>
    <h1>Resume Screener</h1>
    <p>Strict rubric-driven analysis — select a role then upload a resume</p>
</div>
""", unsafe_allow_html=True)

# ── Role selector ────────────────────────────────────────────────────
st.markdown('<p style="color:#444; font-size:0.75rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.3rem">Screening For</p>', unsafe_allow_html=True)
selected_role = st.selectbox(
    label="Role",
    options=ROLES,
    index=0,
    label_visibility="collapsed"
)

# ── File uploader ─────────────────────────────────────────────────────
st.markdown('<p style="color:#444; font-size:0.75rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; margin: 1rem 0 0.3rem">Resume PDF</p>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="Upload Resume PDF",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload a PDF resume to screen"
)

if uploaded:
    st.markdown(f"""
    <div class="upload-indicator">
        <span style="color:#6366f1; font-size:1.1rem">PDF</span>
        <div>
            <div class="fname">{uploaded.name}</div>
            <div class="fmeta">{round(uploaded.size/1024, 1)} KB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"Screen for {selected_role}"):
        sections = ROLE_SKILL_SECTIONS.get(selected_role, ["Skills A", "Skills B", "Skills C"])
        with st.spinner(f"Screening against {selected_role} role..."):
            try:
                full_content = load_pdf(uploaded)
                chain = get_chain()
                result = invoke_with_retry(chain, {
                    "content": full_content,
                    "role":    selected_role,
                    "section1": sections[0],
                    "section2": sections[1],
                    "section3": sections[2],
                })
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()

        # ── Parse fields ──────────────────────────────────────────────
        name            = extract_field(result, "Name")
        role            = extract_field(result, "Preferred Role")
        college         = extract_field(result, "College Name")
        score_raw       = extract_field(result, "Resume Score")
        skills_1        = extract_skills(result, sections[0])
        skills_2        = extract_skills(result, sections[1])
        skills_3        = extract_skills(result, sections[2])
        breakdown       = extract_breakdown(result)
        concerns        = extract_concerns(result)

        try:
            score = int(re.search(r"\d+", score_raw).group())
        except Exception:
            score = 0

        color = score_color(score)

        # ── Overview row ─────────────────────────────────────────────
        st.markdown('<div class="sec-heading">Overview</div>', unsafe_allow_html=True)
        col_score, col_info = st.columns([1, 2], gap="large")

        with col_score:
            st.markdown(f"""
            <div class="score-wrap">
                <div class="score-num" style="color:{color}">{score}</div>
                <div class="score-sub">Score / 100</div>
            </div>
            """, unsafe_allow_html=True)

        with col_info:
            st.markdown(f"""
            <div class="info-card">
                <div class="label">Candidate</div>
                <div class="value">{name}</div>
            </div>
            <div class="info-card">
                <div class="label">Preferred Role</div>
                <div class="value">{role}</div>
            </div>
            <div class="info-card">
                <div class="label">College</div>
                <div class="value">{college}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Score breakdown ───────────────────────────────────────────
        if breakdown:
            rows_html = "".join(f'<div class="bd-row">{md_to_html(line)}</div>' for line in breakdown)
            st.markdown(f"""
            <div class="breakdown">
                <div class="label">Score Breakdown</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        # ── Skills ────────────────────────────────────────────────────
        st.markdown('<div class="sec-heading">Skills</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")

        skill_cols = [
            (c1, sections[0], skills_1),
            (c2, sections[1], skills_2),
            (c3, sections[2], skills_3),
        ]
        for col, label, skills in skill_cols:
            with col:
                chips = "".join(f'<span class="chip">{s}</span>' for s in skills) \
                        or "<span style='color:#333'>None listed</span>"
                st.markdown(f"""
                <div class="skills-card">
                    <div class="label">{label}</div>
                    <div class="chip-wrap">{chips}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Concerns ─────────────────────────────────────────────────
        st.markdown('<div class="sec-heading">Concerns</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="concerns">
            <div class="label">Flagged Issues</div>
            <div>{md_to_html(concerns)}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Raw output ────────────────────────────────────────────────
        with st.expander("Raw LLM output"):
            st.code(result, language="markdown")

else:
    st.markdown("""
    <div style="
        background:#111;
        border:1px dashed #222;
        border-radius:10px;
        padding:3rem 2rem;
        text-align:center;
        margin-top:1rem;
    ">
        <p style="color:#333; font-size:0.85rem; margin:0">Upload a PDF above to begin</p>
    </div>
    """, unsafe_allow_html=True)
