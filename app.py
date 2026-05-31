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
st.set_page_config(page_title="Production Full-Length Resume Engine", page_icon="📝", layout="wide")
st.title("📝 Full-Length Unbiased Resume Optimization Agent")
st.caption("2-Pass Execution Architecture: Guarantees 100% stable, long-form outputs on open-source backends.")

# =====================================================================
# 2. LIGHTWEIGHT DECOUPLED PYDANTIC SCHEMAS
# =====================================================================
class InitialAnalysisMeta(BaseModel):
    extracted_jd_keywords: List[str] = Field(description="Critical skills extracted from the JD.")
    verified_matching_skills: List[str] = Field(description="JD skills verified directly or via platform equivalents (e.g., SageMaker).")
    professional_summary: str = Field(description="Paragraph professional summary tailoring the profile narratives toward MLOps/Infra.")
    technical_skills_matrix: str = Field(description="Comma-separated string matrix mapping tech stack capabilities.")
    calculated_match_score: int = Field(description="ATS match evaluation metrics percentage (0-100).")

class OptimizedBullet(BaseModel):
    original_text: str = Field(description="The source line being evaluated.")
    optimized_text: str = Field(description="The reframed sentence tracking to MLOps, deployment pipelines, or cloud scaling.")
    keywords_integrated: List[str] = Field(description="Keywords applied to this single sentence.")

class FullWorkHistory(BaseModel):
    company: str = Field(description="Company name.")
    role_title: str = Field(description="The updated functional title.")
    duration: str = Field(description="Dates or duration.")
    bullets: List[OptimizedBullet] = Field(description="The complete list of ALL rewritten sentences 1:1.")

class FinalExperiencePayload(BaseModel):
    comprehensive_experience: List[FullWorkHistory] = Field(description="Complete structural array tracking all jobs from the record.")

# =====================================================================
# 3. MULTI-PASS RUNTIME ORCHESTRATION ENGINE
# =====================================================================
def run_stable_resume_agent(job_description: str, raw_resume: str, api_key: str):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0, 
        groq_api_key=api_key
    )
    
    # -----------------------------------------------------------------
    # PASS 1: METADATA & ANALYTICS ANALYSIS
    # -----------------------------------------------------------------
    meta_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical recruiter. Analyze the JD and Resume.\n"
            "Identify target keywords and verify equivalents (AWS SageMaker/SageMaker AI counts as equivalents "
            "for ML Pipelines, Model Deployment, Serving, Governance, and ML Compute).\n"
            "Generate an optimized paragraph professional summary and skills matrix based purely on existing record truths."
        )),
        ("human", "### JD:\n{jd}\n\n### RESUME:\n{resume}")
    ])
    
    meta_chain = meta_prompt | llm.with_structured_output(InitialAnalysisMeta)
    print("[Pass 1/2] Computing baseline analytics and generating meta structures...")
    meta_output = meta_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # -----------------------------------------------------------------
    # PASS 2: COMPREHENSIVE WORK HISTORY REWRITING
    # -----------------------------------------------------------------
    rewriting_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Tech Resume Writer. Your task is to extract and rewrite the candidate's professional work history.\n\n"
            "CRITICAL RULES:\n"
            "1. NO OMISSION: Provide an optimized version for EVERY SINGLE bullet point from the input text. Do not summarize, merge, or shorten the chronology.\n"
            "2. CONTEXTUAL PIVOT: Reframer their DevOps/Cloud experience to focus on ML workflows, Model Deployments (CD), and automated pipelines using their verified tools (SageMaker, Jenkins, Terraform, Docker, Kubernetes).\n"
            "3. NO HALLUCINATION: Do not invent numbers, metrics, or claim they used raw libraries like PyTorch or Airflow if they are missing from their background."
        )),
        ("human", (
            "### TARGET KEYWORDS TO WEAVE IN NATIVELY:\n{verified_skills}\n\n"
            "### WORK HISTORY BLOCK TO REWRITE 1:1:\n{resume_content}"
        ))
    ])
    
    rewriting_chain = rewriting_prompt | llm.with_structured_output(FinalExperiencePayload)
    print("[Pass 2/2] Transforming professional experience blocks 1:1 inside sandbox...")
    experience_output = rewriting_chain.invoke({
        "verified_skills": json.dumps(meta_output.verified_matching_skills),
        "resume_content": raw_resume
    })
    
    return meta_output, experience_output

# =====================================================================
# 4. STREAMLIT LAYOUT INTERFACE
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
            with st.spinner("Executing 2-pass structural context translation engine safely..."):
                try:
                    meta, experience = run_stable_resume_agent(jd_input, resume_input, user_api_key)
                    
                    # Display Metrics
                    st.metric(label="Calculated ATS Match Rating", value=f"{meta.calculated_match_score}%")
                    st.markdown("---")
                    
                    # Render Profile Document
                    st.markdown("### Technical Summary Profile")
                    st.info(f"**Professional Summary:**\n{meta.professional_summary}")
                    
                    st.markdown("### Technical Skills Core Inventory")
                    st.markdown(meta.technical_skills_matrix)
                    
                    st.markdown("### Comprehensive Chronological Work History")
                    for job in experience.comprehensive_experience:
                        st.markdown(f"#### **{job.role_title}** — *{job.company}* ({job.duration})")
                        for idx, bullet in enumerate(job.bullets):
                            st.markdown(f"**{idx+1}.** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Alignment Tags: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'Core Platform Infrastructure'}*")
                    
                    # Log Unmapped Elements
                    unmapped_gaps = [g for g in meta.extracted_jd_keywords if g not in meta.verified_matching_skills]
                    if unmapped_gaps:
                        st.markdown("---")
                        st.markdown("### ⚠️ Blocked Tool Gaps (Sandbox Safety Filter)")
                        st.caption("Omitted from text injection to prevent background misrepresentation:")
                        st.warning(", ".join(unmapped_gaps))
                        
                except Exception as e:
                    st.error(f"An engine runtime error occurred: {str(e)}")
    else:
        st.info("Input your parameters and execute the tool to render your full-length optimized profile layout.")
