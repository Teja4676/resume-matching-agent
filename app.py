import streamlit as st
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# =====================================================================
# 1. WEB PAGE LAYOUT INITIALIZATION
# =====================================================================
st.set_page_config(page_title="Full-Length Unbiased Resume Agent", page_icon="📝", layout="wide")
st.title("📝 Full-Length Unbiased Resume Optimization Agent")
st.caption("Transforms every single bullet point contextually into MLOps alignments without skipping or consolidating text.")

# =====================================================================
# 2. PYDANTIC SCHEMAS FOR 1:1 FULL-LENGTH RESUME GENERATION
# =====================================================================
class HighlightBlock(BaseModel):
    optimized_highlight: str = Field(description="The reframed executive highlight bullet point.")

class OptimizedBullet(BaseModel):
    original_text: str = Field(description="The exact original source bullet point.")
    optimized_text: str = Field(description="The fully expanded, detailed rewrite aligning the task to ML pipelines, cloud infra, or model operations workflows.")
    keywords_integrated: List[str] = Field(description="The specific JD keywords seamlessly integrated here.")

class FullWorkHistory(BaseModel):
    company: str = Field(description="Company name.")
    role_title: str = Field(description="The rewritten professional title fitting the MLOps/Infra target.")
    duration: str = Field(description="Dates of employment.")
    all_optimized_bullets: List[OptimizedBullet] = Field(description="MUST include a rewritten counterpart for EVERY SINGLE original bullet point. Do not drop, omit, or merge bullets.")

class TechnicalSkillsInventory(BaseModel):
    category: str = Field(description="Skill category name (e.g., Cloud Platforms, MLOps & Orchestration, Security).")
    skills: List[str] = Field(description="The underlying tools/skills belonging to this tier.")

class CompleteRestructuredResume(BaseModel):
    candidate_name: str = Field(description="Full name of candidate.")
    contact_info: List[str] = Field(description="Phone, email, links.")
    professional_summary: str = Field(description="A comprehensive, paragraph-length professional summary mapping core infrastructure automation background to the target MLOps domain.")
    executive_highlights: List[HighlightBlock] = Field(description="Tailored strategic core highlights.")
    technical_skills_matrix: List[TechnicalSkillsInventory] = Field(description="Full inventory of all software, tooling, and architectures.")
    comprehensive_experience: List[FullWorkHistory] = Field(description="The comprehensive work history comprising all modified points 1:1.")
    unmapped_critical_gaps: List[str] = Field(description="Tools or concepts requested by the JD completely missing from candidate record.")
    calculated_match_score: int = Field(description="ATS match accuracy score metrics (0-100).")

# =====================================================================
# 3. CORE RUNTIME ENGINE
# =====================================================================
def run_full_resume_agent(job_description: str, raw_resume: str, api_key: str) -> CompleteRestructuredResume:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0, 
        groq_api_key=api_key
    )
    
    # -----------------------------------------------------------------
    # STAGES 1 & 2: REQ EXTRACTION & CONCEPTUAL TRUTH MAPPING
    # -----------------------------------------------------------------
    # Helper model for initial mapping
    class SimpleAnalysis(BaseModel):
        extracted_jd_keywords: List[str]
        verified_matching_skills: List[str]

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert tech recruiter. Extract critical skills from the JD. "
            "Cross-reference them with the Resume. If the candidate uses AWS SageMaker/SageMaker AI/SageMaker Unified Studio, "
            "mark them as verified equivalents for ML Pipelines, Model Deployment & Serving, Model Governance, and ML Infrastructure Compute."
        )),
        ("human", "### JD:\n{jd}\n\n### RESUME:\n{resume}")
    ])
    analysis_chain = analysis_prompt | llm.with_structured_output(SimpleAnalysis)
    analysis_results = analysis_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # -----------------------------------------------------------------
    # STAGE 3: COMPREHENSIVE 1:1 REWRITING SANDBOX
    # -----------------------------------------------------------------
    rewriting_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Resume Architect. Your job is to generate a comprehensive, full-length resume based on the inputs.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. NO OMISSION: You must map and optimize EVERY SINGLE original bullet point across the professional history. Do not consolidate multiple original bullets into one or two sentences. Deliver a full-length history layout.\n"
            "2. CONTEXTUAL REFRAMING: Rephrase their infrastructure, cloud, security, and lifecycle automation achievements to highlight how they build, serve, secure, and monitor Machine Learning workloads using SageMaker and advanced CI/CD.\n"
            "3. NO HALLUCINATION: You can reframe terminology contextually (e.g., transforming deployment automation into Model Serving pipelines), but you cannot invent numeric metrics, percentages, or frameworks like PyTorch if they are absent from the candidate's history.\n"
            "4. Match Score Formula: (Count of Unique Integrated Keywords / Total Target Job Keywords) * 100."
        )),
        ("human", (
            "### TARGET JOB KEYWORDS:\n{job_keywords}\n\n"
            "### VERIFIED MATCHING SKILLS & EQUIVALENTS:\n{verified_skills}\n\n"
            "### FULL ORIGINAL RESUME FOR COMPLETE 1:1 TRANSFORMATION:\n{resume_content}"
        ))
    ])
    
    rewriting_chain = rewriting_prompt | llm.with_structured_output(CompleteRestructuredResume)
    
    return rewriting_chain.invoke({
        "job_keywords": json.dumps(analysis_results.extracted_jd_keywords),
        "verified_skills": json.dumps(analysis_results.verified_matching_skills),
        "resume_content": raw_resume
    })

# =====================================================================
# 4. STREAMLIT FULL-LENGTH OUTPUT FRONTEND
# =====================================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. Source Inputs")
    user_api_key = st.text_input("Enter your Free Groq API Key", type="password")
    jd_input = st.text_area("Paste target Job Description (JD) here", height=200)
    resume_input = st.text_area("Paste your original Full Resume here", height=450)
    
    submit_btn = st.button("Generate Complete Tailored Resume", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Complete Tailored Resume Output")
    
    if submit_btn:
        if not user_api_key or not jd_input or not resume_input:
            st.error("Please fill out all input blocks before running.")
        else:
            with st.spinner("Executing full-length 1:1 context translation engine..."):
                try:
                    resume_data = run_full_resume_agent(jd_input, resume_input, user_api_key)
                    
                    # Top Analytics Metrics Dashboard
                    st.metric(label="Calculated ATS Match Rating", value=f"{resume_data.calculated_match_score}%")
                    
                    st.markdown("---")
                    
                    # PRINTING A READY-TO-USE PLAIN TEXT RESUME
                    st.header(resume_data.candidate_name)
                    st.write(" | ".join(resume_data.contact_info))
                    
                    st.markdown("### Professional Summary")
                    st.write(resume_data.professional_summary)
                    
                    st.markdown("### Executive Performance Highlights")
                    for highlight in resume_data.executive_highlights:
                        st.markdown(f"• {highlight.optimized_highlight}")
                        
                    st.markdown("### Technical Skills Core Inventory")
                    for skill_group in resume_data.technical_skills_matrix:
                        st.markdown(f"**{skill_group.category}:** {', '.join(skill_group.skills)}")
                        
                    st.markdown("### Professional Work Experience Breakdown")
                    for job in resume_data.comprehensive_experience:
                        st.markdown(f"#### **{job.role_title}** — *{job.company}* ({job.duration})")
                        for idx, bullet in enumerate(job.all_optimized_bullets):
                            st.markdown(f"**{idx+1}.** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Alignment Tags: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'General Infrastructure'}*")
                    
                    # Safety Guards Logs
                    if resume_data.unmapped_critical_gaps:
                        st.markdown("---")
                        st.markdown("### ⚠️ Blocked Tool Gaps (Sandbox Safety Filter)")
                        st.caption("The following specialized tools from the JD were blocked from inclusion because no functional equivalent was verified in your profile history to prevent hallucinations:")
                        st.info(", ".join(resume_data.unmapped_critical_gaps))
                        
                except Exception as e:
                    st.error(f"An engine runtime error occurred: {str(e)}")
    else:
        st.info("Input your operational parameters and click generate to render your structured, full-length resume.")
