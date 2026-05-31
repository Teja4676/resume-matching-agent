import streamlit as st
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# =====================================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Stabilized Full-Length Resume Engine", page_icon="📝", layout="wide")
st.title("📝 Full-Length Unbiased Resume Optimization Agent")
st.caption("Stabilized 1:1 Map-and-Transform Engine for Complete Career Timelines.")

# =====================================================================
# 2. STABILIZED SCHEMAS FOR FAULT-TOLERANT PARSING
# =====================================================================
class OptimizedBullet(BaseModel):
    original_text: str = Field(description="The exact original source bullet point from the input resume.")
    optimized_text: str = Field(description="The rewritten bullet aligning the task natively to MLOps, cloud infrastructure, or automated pipelines.")
    keywords_integrated: List[str] = Field(description="The specific target keywords integrated into this line.")

class OptimizedWorkHistory(BaseModel):
    company: str = Field(description="Company or Organization name.")
    role_title: str = Field(description="The professional title contextually aligned for the target domain.")
    duration: str = Field(description="Dates or duration of employment.")
    bullets: List[OptimizedBullet] = Field(description="List of EVERY rewritten bullet point for this company. Maintain a strict 1:1 ratio.")

class CompleteRestructuredResume(BaseModel):
    candidate_name: str = Field(description="Full name of candidate.")
    professional_summary: str = Field(description="A comprehensive professional summary paragraph linking infrastructure automation to production MLOps engineering.")
    technical_skills_matrix: str = Field(description="A clean, single string or comma-separated markdown representation of optimized tools and platforms.")
    comprehensive_experience: List[OptimizedWorkHistory] = Field(description="The full array of chronological career history rewritten 1:1.")
    unmapped_critical_gaps: List[str] = Field(description="Target requirements from the JD completely missing from candidate's profile data.")
    calculated_match_score: int = Field(description="ATS match score percentage (0-100).")

# =====================================================================
# 3. CORE RUNTIME ENGINE
# =====================================================================
def run_full_resume_agent(job_description: str, raw_resume: str, api_key: str) -> CompleteRestructuredResume:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0, 
        groq_api_key=api_key
    )
    
    # --- STAGES 1 & 2: REQUIREMENTS EXTRACTION & EQUIVALENCY MAPPING ---
    class SimpleAnalysis(BaseModel):
        extracted_jd_keywords: List[str]
        verified_matching_skills: List[str]

    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert tech recruiter. Extract critical skills from the JD. "
            "Cross-reference them with the Resume. If the candidate has extensive experience with AWS SageMaker, SageMaker AI, or SageMaker Unified Studio, "
            "always count them as verified equivalents for ML Pipelines, Model Deployment, Serving, Model Governance, and ML Infrastructure Compute."
        )),
        ("human", "### JD:\n{jd}\n\n### RESUME:\n{resume}")
    ])
    analysis_chain = analysis_prompt | llm.with_structured_output(SimpleAnalysis)
    analysis_results = analysis_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # --- STAGE 3: FULL 1:1 RESUME TRANSLATION SANDBOX ---
    rewriting_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Resume Architect. Your job is to output a comprehensive, full-length resume matching the schema.\n\n"
            "CRITICAL RULES:\n"
            "1. NO OMISSION: You must map and rewrite EVERY SINGLE original bullet point across the professional history. Do not drop, skip, or merge any bullets. Provide a full-length layout.\n"
            "2. CONTEXTUAL REFRAMING: Pivot their infrastructure, cloud scaling, validation, and automation achievements to highlight how they build, serve, secure, and monitor production Machine Learning workloads using SageMaker and advanced CI/CD.\n"
            "3. NO HALLUCINATION: You must never invent numbers, metrics, or technologies (like PyTorch or Airflow) if they are absent from the candidate's history. Rely heavily on their existing tools (Jenkins, Terraform, SageMaker, Kubernetes, Docker).\n"
            "4. Match Score Formula: (Count of Unique Integrated Keywords / Total Target Job Keywords) * 100."
        )),
        ("human", (
            "### TARGET JOB KEYWORDS:\n{job_keywords}\n\n"
            "### VERIFIED MATCHING SKILLS & EQUIVALENTS:\n{verified_skills}\n\n"
            "### FULL ORIGINAL RESUME FOR COMPLETE TRANSFORMATION:\n{resume_content}"
        ))
    ])
    
    rewriting_chain = rewriting_prompt | llm.with_structured_output(CompleteRestructuredResume)
    
    return rewriting_chain.invoke({
        "job_keywords": json.dumps(analysis_results.extracted_jd_keywords),
        "verified_skills": json.dumps(analysis_results.verified_matching_skills),
        "resume_content": raw_resume
    })

# =====================================================================
# 4. STREAMLIT DISPLAY INTERFACE
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
            st.error("Please fill out all input fields before running.")
        else:
            with st.spinner("Executing full-length structural context translation engine..."):
                try:
                    resume_data = run_full_resume_agent(jd_input, resume_input, user_api_key)
                    
                    # Display Match Rating
                    st.metric(label="Calculated ATS Match Rating", value=f"{resume_data.calculated_match_score}%")
                    st.markdown("---")
                    
                    # Render Profile Header
                    st.header(resume_data.candidate_name)
                    
                    st.markdown("### Professional Summary")
                    st.write(resume_data.professional_summary)
                    
                    st.markdown("### Technical Skills Matrix")
                    st.markdown(resume_data.technical_skills_matrix)
                    
                    st.markdown("### Professional Work Experience Breakdown")
                    for job in resume_data.comprehensive_experience:
                        st.markdown(f"#### **{job.role_title}** — *{job.company}* ({job.duration})")
                        for idx, bullet in enumerate(job.bullets):
                            st.markdown(f"**{idx+1}.** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Alignment Tags: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'Core Platform Infrastructure'}*")
                    
                    # Log Blocked Gaps
                    if resume_data.unmapped_critical_gaps:
                        st.markdown("---")
                        st.markdown("### ⚠️ Blocked Tool Gaps (Sandbox Safety Filter)")
                        st.caption("The following specialized tools from the JD were blocked from inclusion because no functional equivalent was verified in your profile history to prevent hallucinations:")
                        st.info(", ".join(resume_data.unmapped_critical_gaps))
                        
                except Exception as e:
                    st.error(f"An engine runtime error occurred: {str(e)}")
    else:
        st.info("Input your operational parameters and click generate to render your structured, full-length resume.")
