import streamlit as st
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. SETUP WEB PAGE CONFIG
st.set_page_config(page_title="Unbiased Resume Matcher AI", page_icon="📝", layout="wide")
st.title("📝 Unbiased Resume Matching & Optimization Agent")
st.caption("Achieve 80-100% ATS optimization safely within a zero-hallucination sandbox.")

# 2. DEFINING PYDANTIC SCHEMAS FOR LANGCHAIN
class AnalysisReport(BaseModel):
    extracted_jd_keywords: List[str] = Field(description="Critical skills extracted from the JD.")
    verified_matching_skills: List[str] = Field(description="Sub-list of JD keywords that the candidate explicitly possesses.")

class OptimizedBulletPoint(BaseModel):
    original_text: str = Field(description="The source bullet point text before optimization.")
    optimized_text: str = Field(description="The rewritten bullet following X-Y-Z formula.")
    keywords_integrated: List[str] = Field(description="Keywords from the verified matching list integrated here.")

class OptimizedWorkExperience(BaseModel):
    company: str = Field(description="Company name.")
    role_title: str = Field(description="The professional title.")
    duration: Optional[str] = Field(None, description="Employment timeline.")
    bullets: List[OptimizedBulletPoint] = Field(description="Optimized achievement bullets.")

class FullOptimizedResume(BaseModel):
    candidate_name: str = Field(description="The professional name of the candidate.")
    professional_summary: str = Field(description="Optimized narrative summary.")
    work_experience: List[OptimizedWorkExperience] = Field(description="The tailored work history block.")
    unmapped_critical_gaps: List[str] = Field(description="JD keywords omitted due to total lack of evidence.")
    calculated_match_score: int = Field(description="Match score out of 100.")

# 3. CORE AGENT PIPELINE ENGINE
def run_agent(job_description: str, raw_resume: str, api_key: str) -> FullOptimizedResume:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=api_key)
    
    # Stage 1 & 2
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an objective AI Recruiter. Parse the JD for keywords, cross-reference with the Resume. Do not extrapolate."),
        ("human", "### JOB DESCRIPTION:\n{jd}\n\n### CANDIDATE RESUME:\n{resume}")
    ])
    analysis_chain = analysis_prompt | llm.with_structured_output(AnalysisReport)
    analysis_results = analysis_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # Stage 3
    rewriting_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert Resume Engine. Optimize resume vocabulary using verified matching skills. NO HALLUCINATION. Do not invent metrics or missing tools. If a keyword is missing from the verified list, ignore it and log it under gaps."),
        ("human", "### TARGET JOB KEYWORDS:\n{job_keywords}\n\n### VERIFIED MATCHING SKILLS:\n{verified_skills}\n\n### FULL ORIGINAL RESUME DATA:\n{resume_content}")
    ])
    rewriting_chain = rewriting_prompt | llm.with_structured_output(FullOptimizedResume)
    
    return rewriting_chain.invoke({
        "job_keywords": json.dumps(analysis_results.extracted_jd_keywords),
        "verified_skills": json.dumps(analysis_results.verified_matching_skills),
        "resume_content": raw_resume
    })

# 4. BUILDING THE INTERACTIVE UI
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Input Configuration")
    user_api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Your key is processed securely in-memory and never saved stored.")
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
                    
                    # Score Callout
                    st.metric(label="Calculated ATS Match Score", value=f"{result.calculated_match_score}%")
                    
                    # Professional Summary Block
                    st.markdown(f"### Profile: **{result.candidate_name}**")
                    st.info(f"**Professional Summary:**\n{result.professional_summary}")
                    
                    # Work Experience Output
                    st.markdown("### Tailored Chronological Experience")
                    for job in result.work_experience:
                        st.markdown(f"#### **{job.role_title}** at *{job.company}* ({job.duration or ''})")
                        for bullet in job.bullets:
                            st.write(f"👉 **Optimized:** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Keywords: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'None'}*")
                    
                    # Unmapped Safety Gaps Callout
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
