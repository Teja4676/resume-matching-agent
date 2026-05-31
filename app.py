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
st.set_page_config(page_title="Unbiased Resume Matcher AI", page_icon="📝", layout="wide")
st.title("📝 Advanced MLOps/DevSecOps Resume Matching Agent")
st.caption("Enhanced Semantic Matching Engine - Bridges the gap between cloud ML platforms and MLOps tools.")

# =====================================================================
# 2. PYDANTIC SCHEMAS FOR STRUCTURED DATA VALIDATION
# =====================================================================
class AnalysisReport(BaseModel):
    extracted_jd_keywords: List[str] = Field(description="Critical skills and concepts extracted from the JD.")
    verified_matching_skills: List[str] = Field(description="Sub-list of JD keywords that the candidate explicitly possesses OR has direct cloud platform equivalents for (e.g., SageMaker for MLOps).")

class OptimizedBulletPoint(BaseModel):
    original_text: str = Field(description="The source bullet point text before optimization.")
    optimized_text: str = Field(description="The rewritten bullet following X-Y-Z formula, emphasizing the target role context.")
    keywords_integrated: List[str] = Field(description="Keywords from the verified matching list integrated here.")

class OptimizedWorkExperience(BaseModel):
    company: str = Field(description="Company name.")
    role_title: str = Field(description="The professional title optimized for the target domain context.")
    duration: Optional[str] = Field(None, description="Employment timeline.")
    bullets: List[OptimizedBulletPoint] = Field(description="Optimized achievement bullets.")

class FullOptimizedResume(BaseModel):
    candidate_name: str = Field(description="The professional name of the candidate.")
    professional_summary: str = Field(description="Optimized narrative summary bridging existing skills to target requirements.")
    work_experience: List[OptimizedWorkExperience] = Field(description="The tailored work history block.")
    unmapped_critical_gaps: List[str] = Field(description="JD keywords omitted due to total lack of evidence.")
    calculated_match_score: int = Field(description="Match score out of 100.")

# =====================================================================
# 3. CORE 3-STAGE AGENT RUNTIME PIPELINE
# =====================================================================
def run_agent(job_description: str, raw_resume: str, api_key: str) -> FullOptimizedResume:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0, 
        groq_api_key=api_key
    )
    
    # --- STAGES 1 & 2: CONCEPTUAL REQ EXTRACTION & TRUTH MAPPING ---
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert AI Recruiter specializing in MLOps, DevOps, and Data Science infrastructure.\n"
            "Your job is to identify core keywords in the JD and evaluate if the candidate has them OR their direct engineering equivalents.\n\n"
            "CRITICAL EQUIVALENCY RULES:\n"
            "If the candidate has extensive experience with AWS SageMaker, SageMaker AI, and SageMaker Unified Studio, they possess valid platform equivalents for:\n"
            "- ML Pipelines / Orchestration (equivalent to Kubeflow / Airflow workflows)\n"
            "- Model Deployment & Serving / Model Registry (equivalent to MLflow / Weights & Biases tracking)\n"
            "- Machine Learning Infrastructure / Compute\n\n"
            "If an equivalent exists, include the JD keyword in the 'verified_matching_skills' array so the rewriter can optimize the phrasing contextually."
        )),
        ("human", "### JOB DESCRIPTION:\n{jd}\n\n### CANDIDATE RESUME:\n{resume}")
    ])
    analysis_chain = analysis_prompt | llm.with_structured_output(AnalysisReport)
    analysis_results = analysis_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # --- STAGE 3: CONTEXTUAL REWRITING SANDBOX ---
    rewriting_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Resume Engine. Your goal is to pivot a candidate's DevOps/DevSecOps experience into the MLOps context "
            "using their verified AWS SageMaker and automation background.\n\n"
            "CRITICAL SANDBOX RULES:\n"
            "1. NO HALLUCINATION. Do not invent metrics, projects, or say they wrote raw PyTorch models if they didn't. \n"
            "2. CONTEXTUAL TRANSLATION: Reframe their deployment, automation, and infrastructure engineering achievements to highlight how they support ML Pipelines, Model Deployment (CD), and ML Infrastructure using SageMaker and CI/CD.\n"
            "3. Only use keywords that are present in the 'Verified Matching Skills' list.\n"
            "4. If a keyword is completely unmapped and lacks any equivalent, log it under gaps."
        )),
        ("human", (
            "### TARGET JOB KEYWORDS:\n{job_keywords}\n\n"
            "### VERIFIED MATCHING SKILLS:\n{verified_skills}\n\n"
            "### FULL ORIGINAL RESUME DATA:\n{resume_content}"
        ))
    ])
    rewriting_chain = rewriting_prompt | llm.with_structured_output(FullOptimizedResume)
    
    return rewriting_chain.invoke({
        "job_keywords": json.dumps(analysis_results.extracted_jd_keywords),
        "verified_skills": json.dumps(analysis_results.verified_matching_skills),
        "resume_content": raw_resume
    })

# =====================================================================
# 4. STREAMLIT FRONTEND LAYOUT
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input Configuration")
    user_api_key = st.text_input("Enter your Free Groq API Key", type="password")
    jd_input = st.text_area("Paste target Job Description (JD) here", height=200)
    resume_input = st.text_area("Paste your original Resume here", height=300)
    
    submit_btn = st.button("Run Sandbox Optimization Engine", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Optimization Output & Analytics")
    
    if submit_btn:
        if not user_api_key or not jd_input or not resume_input:
            st.error("Please fill out your API Key, Job Description, and Resume before running!")
        else:
            with st.spinner("Processing 3-Stage Pipeline (Analyzing, Cross-referencing, Rewriting safely)..."):
                try:
                    result = run_agent(jd_input, resume_input, user_api_key)
                    
                    st.metric(label="Calculated ATS Match Score", value=f"{result.calculated_match_score}%")
                    st.markdown(f"### Profile: **{result.candidate_name}**")
                    st.info(f"**Professional Summary:**\n{result.professional_summary}")
                    
                    st.markdown("### Tailored Chronological Experience")
                    for job in result.work_experience:
                        st.markdown(f"#### **{job.role_title}** at *{job.company}* ({job.duration or ''})")
                        for bullet in job.bullets:
                            st.write(f"👉 **Optimized:** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Keywords: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'None'}*")
                    
                    if result.unmapped_critical_gaps:
                        st.markdown("---")
                        st.markdown("### ⚠️ Blocked Gaps (Sandbox Safety Filter)")
                        st.warning("The following core requirements from the Job Description were intentionally excluded to prevent resume exaggeration/hallucinations:")
                        for gap in result.unmapped_critical_gaps:
                            st.markdown(f"- `{gap}`")
                            
                except Exception as e:
                    st.error(f"An engine runtime error occurred: {str(e)}")
    else:
        st.write("Provide inputs on the left pane and execute the engine to populate analysis.")
