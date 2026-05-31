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
st.set_page_config(page_title="Cellular Resume Optimization Engine", page_icon="📝", layout="wide")
st.title("📝 Full-Length Unbiased Resume Optimization Agent")
st.caption("Cellular Mapping Engine: Processes work history sequentially to guarantee 100% stable, long-form output strings.")

# =====================================================================
# 2. STRIPPED DOWN CELLULAR SCHEMAS FOR PERFECT RECOVERY
# =====================================================================
class InitialAnalysisMeta(BaseModel):
    extracted_jd_keywords: List[str] = Field(description="Critical skills extracted from the JD.")
    verified_matching_skills: List[str] = Field(description="JD skills verified directly or via equivalents like SageMaker.")
    professional_summary: str = Field(description="Paragraph professional summary tailoring the narrative toward MLOps/Infra.")
    technical_skills_matrix: str = Field(description="Comma-separated string mapping technical capabilities.")
    calculated_match_score: int = Field(description="ATS match evaluation rating metrics percentage (0-100).")

class OptimizedBullet(BaseModel):
    original_text: str = Field(description="The source line being evaluated.")
    optimized_text: str = Field(description="The reframed sentence tracking specifically to the target domain requirements.")
    keywords_integrated: List[str] = Field(description="Keywords applied to this single line.")

class SingleCompanyPayload(BaseModel):
    role_title: str = Field(description="The updated functional title contextually adapted for the target.")
    bullets: List[OptimizedBullet] = Field(description="The complete list of rewritten sentences matching the original inputs 1:1.")

# =====================================================================
# 3. INTERACTIVE RESUME PARSING UTILITIES
# =====================================================================
def parse_resume_to_blocks(raw_text: str) -> List[dict]:
    """
    Helper parsing utility to break down work history strings into structured 
    blocks to prevent token limit bottlenecks.
    """
    company_blocks = []
    
    # Simple semantic splitting markers based on standard resume structures
    markers = ["Mphasis", "CoreBridge", "Infosys", "Renault Nissan", "Innobox Systems"]
    lines = raw_text.split("\n")
    
    current_company = None
    current_duration = "Unspecified"
    current_lines = []
    
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
            
        found_marker = False
        for marker in markers:
            if marker.lower() in cleaned.lower() and ("present" in cleaned.lower() or any(yr in cleaned for yr in ["2018", "2021", "2022", "2023", "2025"])):
                if current_company:
                    company_blocks.append({"company": current_company, "duration": current_duration, "content": "\n".join(current_lines)})
                current_company = marker
                current_duration = cleaned.replace(marker, "").strip(" \t,-—–")
                current_lines = []
                found_marker = True
                break
        
        if not found_marker and current_company:
            current_lines.append(cleaned)
            
    if current_company:
        company_blocks.append({"company": current_company, "duration": current_duration, "content": "\n".join(current_lines)})
        
    return company_blocks

# =====================================================================
# 4. CELLULAR RUNTIME ORCHESTRATION ENGINE
# =====================================================================
def run_cellular_agent(job_description: str, raw_resume: str, api_key: str):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=api_key)
    
    # -----------------------------------------------------------------
    # PASS 1: METADATA EXTRACTION & ATS BASELINE ANALYSIS
    # -----------------------------------------------------------------
    meta_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert technical recruiter. Analyze the JD and full Resume.\n"
            "Identify target keywords and verify equivalents (AWS SageMaker/SageMaker AI counts as equivalents "
            "for ML Pipelines, Model Deployment, Serving, Governance, and ML Compute).\n"
            "Generate an optimized paragraph professional summary and markdown tech skills matrix string."
        )),
        ("human", "### JD:\n{jd}\n\n### FULL RESUME:\n{resume}")
    ])
    meta_chain = meta_prompt | llm.with_structured_output(InitialAnalysisMeta)
    meta_output = meta_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # -----------------------------------------------------------------
    # PASS 2: CELLULAR ITERATION OVER INDIVIDUAL COMPANIES
    # -----------------------------------------------------------------
    company_inputs = parse_resume_to_blocks(raw_resume)
    processed_history = []
    
    cell_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert tech resume optimization writer. You are rewriting experience data for a single company profile.\n"
            "CRITICAL ATOMIC RULES:\n"
            "1. NO OMISSION: Optimize and rewrite EVERY SINGLE bullet point found in the input text block. Do not drop or combine sentences.\n"
            "2. TARGET REFRAMING: Align their technical achievements to highlight building, serving, or monitoring production Machine Learning workloads using their verified stack (SageMaker, Jenkins, Terraform, Docker, Kubernetes).\n"
            "3. NO HALLUCINATION: Never invent synthetic percentages or tools. Stick strictly to background facts."
        )),
        ("human", (
            "### TARGET KEYWORDS TO WEAVE IN:\n{verified_skills}\n\n"
            "### SINGLE COMPANY TARGET TRACK:\nCompany: {company_name}\nContent:\n{company_content}"
        ))
    ])
    cell_chain = cell_prompt | llm.with_structured_output(SingleCompanyPayload)
    
    # Process each company independently to prevent total token limits overflow
    for block in company_inputs:
        cell_output = cell_chain.invoke({
            "verified_skills": json.dumps(meta_output.verified_matching_skills),
            "company_name": block["company"],
            "company_content": block["content"]
        })
        processed_history.append({
            "company": block["company"],
            "role_title": cell_output.role_title,
            "duration": block["duration"],
            "bullets": cell_output.bullets
        })
        
    return meta_output, processed_history

# =====================================================================
# 5. STREAMLIT LAYOUT INTERFACE
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
            with st.spinner("Executing cell-isolated structural context optimization..."):
                try:
                    meta, comprehensive_history = run_cellular_agent(jd_input, resume_input, user_api_key)
                    
                    # Display Metrics
                    st.metric(label="Calculated ATS Match Rating", value=f"{meta.calculated_match_score}%")
                    st.markdown("---")
                    
                    # Render Document Profile
                    st.header("P. Tharun Teja")
                    st.markdown("### Professional Summary")
                    st.write(meta.professional_summary)
                    
                    st.markdown("### Technical Skills Core Matrix")
                    st.markdown(meta.technical_skills_matrix)
                    
                    st.markdown("### Detailed Chronological Experience Breakdown")
                    for job in comprehensive_history:
                        st.markdown(f"#### **{job['role_title']}** — *{job['company']}* ({job['duration']})")
                        for idx, bullet in enumerate(job["bullets"]):
                            st.markdown(f"**{idx+1}.** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Tags: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'Platform Operations Infrastructure'}*")
                    
                    # Log Unmapped Elements
                    unmapped_gaps = [g for g in meta.extracted_jd_keywords if g not in meta.verified_matching_skills]
                    if unmapped_gaps:
                        st.markdown("---")
                        st.markdown("### ⚠️ Blocked Tool Gaps (Sandbox Safety Filter)")
                        st.info(", ".join(unmapped_gaps))
                        
                except Exception as e:
                    st.error(f"An engine runtime error occurred: {str(e)}")
    else:
        st.info("Input parameters and execute tool to run the cellular orchestration loop framework safely.")
