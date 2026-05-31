import streamlit as st
import os
import json
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# =====================================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================================
st.set_page_config(page_title="Universal Unbiased Resume Engine", page_icon="📝", layout="wide")
st.title("📝 Universal Unbiased Resume Optimization Agent")
st.caption("Dynamic Architecture: Extracts and freezes original roles at runtime to prevent profile tampering across any resume.")

# =====================================================================
# 2. DYNAMIC PYDANTIC SCHEMAS FOR MULTI-USER COGNITION
# =====================================================================
class ParsedEmployment(BaseModel):
    company_name: str = Field(description="The exact name of the company/organization as written in the resume.")
    original_title: str = Field(description="The exact original professional title held at this company as written in the resume.")

class InitialAnalysisMeta(BaseModel):
    extracted_jd_keywords: List[str] = Field(description="Critical skills extracted from the JD.")
    verified_matching_skills: List[str] = Field(description="JD skills verified directly or via equivalents in the candidate profile.")
    detected_employment_history: List[ParsedEmployment] = Field(description="List of all detected companies and original titles parsed from the resume.")
    professional_summary: str = Field(description="Paragraph professional summary tailoring the narrative context toward the target role without inventing fake history.")
    technical_skills_matrix: str = Field(description="Markdown string representation of the candidate's core technical stack.")
    calculated_match_score: int = Field(description="ATS match evaluation rating metrics percentage (0-100).")

class OptimizedBullet(BaseModel):
    original_text: str = Field(description="The source bullet line being evaluated.")
    optimized_text: str = Field(description="The reframed sentence tracking specifically to the target requirements without adding fake tools.")
    keywords_integrated: List[str] = Field(description="Keywords applied to this single line.")

class SingleCompanyPayload(BaseModel):
    bullets: List[OptimizedBullet] = Field(description="The complete list of rewritten sentences matching the original inputs 1:1.")

# =====================================================================
# 3. DYNAMIC SEMANTIC RESUME BLOCK SPLITTER
# =====================================================================
def parse_resume_into_dynamic_blocks(raw_text: str, detected_companies: List[str]) -> List[dict]:
    """
    Dynamically slices the resume text into individual company blocks using the 
    companies detected by the LLM in Pass 1. Completely removes hardcoding.
    """
    company_blocks = []
    lines = raw_text.split("\n")
    
    current_company = None
    current_lines = []
    
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
            
        found_marker = False
        for company in detected_companies:
            # Check if the line references a detected company name safely
            if company.lower() in cleaned.lower() and any(char.isdigit() for char in cleaned):
                if current_company:
                    company_blocks.append({"company": current_company, "content": "\n".join(current_lines)})
                current_company = company
                current_lines = []
                found_marker = True
                break
                
        if not found_marker and current_company:
            current_lines.append(cleaned)
            
    if current_company:
        company_blocks.append({"company": current_company, "content": "\n".join(current_lines)})
        
    return company_blocks

# =====================================================================
# 4. RUNTIME ORCHESTRATION ENGINE (UNIVERSAL & UNBIASED)
# =====================================================================
def run_universal_agent(job_description: str, raw_resume: str, api_key: str):
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, groq_api_key=api_key)
    
    # -----------------------------------------------------------------
    # PASS 1: DYNAMIC SCHEMA AND METADATA EXTRACTION
    # -----------------------------------------------------------------
    meta_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an objective expert technical recruiter. Analyze the provided JD and Resume.\n"
            "1. Extract core target keywords from the JD.\n"
            "2. Scrape the resume and extract a clean list of all companies and the exact original job titles held. DO NOT alter these titles based on the JD.\n"
            "3. Generate an honest narrative summary and skills matrix based purely on existing, verifiable resume facts."
        )),
        ("human", "### TARGET JOB DESCRIPTION:\n{jd}\n\n### CANDIDATE RESUME:\n{resume}")
    ])
    meta_chain = meta_prompt | llm.with_structured_output(InitialAnalysisMeta)
    st.write("⚙️ [Pass 1/2] Dynamically analyzing resume architecture and freezing original job titles...")
    meta_output = meta_chain.invoke({"jd": job_description, "resume": raw_resume})
    
    # Map the dynamically discovered companies into a frozen runtime dictionary
    dynamic_title_map = {emp.company_name: emp.original_title for emp in meta_output.detected_employment_history}
    detected_company_names = list(dynamic_title_map.keys())
    
    # -----------------------------------------------------------------
    # PASS 2: CELLULAR REWRITING USING DISCOVERED META VARIABLES
    # -----------------------------------------------------------------
    company_inputs = parse_resume_into_dynamic_blocks(raw_resume, detected_company_names)
    processed_history = []
    
    cell_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a strict, unbiased resume optimization writer. You are processing experience data for a single company profile.\n\n"
            "CRITICAL UNBIASED BOUNDARIES:\n"
            "1. NO TITLE MANIPULATION: You are forbidden from defining or changing the job title. It is handled as an immutable system variable.\n"
            "2. NO SYSTEMATIC FABRICATION: Do not force target context tags (like ML, MLOps, or specific data frameworks) into historical roles where the candidate was performing traditional engineering, testing, or platform administration.\n"
            "3. 1:1 VOCABULARY ALIGNMENT: Optimize the technical vocabulary of the existing achievements to match the target keywords (e.g., aligning automated pipeline naming conventions) without altering historical core responsibilities or inventing metrics."
        )),
        ("human", (
            "### TARGET KEYWORDS TO ALIGN:\n{verified_skills}\n\n"
            "### TARGET PROFILE BLOCK TO TRANSFORM 1:1:\nCompany: {company_name}\nContent:\n{company_content}"
        ))
    ])
    cell_chain = cell_prompt | llm.with_structured_output(SingleCompanyPayload)
    
    st.write(f"⚙️ [Pass 2/2] Running individual optimization loops across {len(company_inputs)} dynamically parsed work blocks...")
    for block in company_inputs:
        # Retrieve the dynamically parsed original title
        frozen_title = dynamic_title_map.get(block["company"], "Technical Specialist")
        
        cell_output = cell_chain.invoke({
            "verified_skills": json.dumps(meta_output.verified_matching_skills),
            "company_name": block["company"],
            "company_content": block["content"]
        })
        processed_history.append({
            "company": block["company"],
            "role_title": frozen_title,
            "bullets": cell_output.bullets
        })
        
    return meta_output, processed_history

# =====================================================================
# 5. STREAMLIT APPLICATION FRONTEND
# =====================================================================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. Source Inputs")
    user_api_key = st.text_input("Enter your Free Groq API Key", type="password")
    jd_input = st.text_area("Paste target Job Description (JD) here", height=200)
    resume_input = st.text_area("Paste your original Full Resume here", height=450)
    
    submit_btn = st.button("Generate Tailored Resume", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Complete Tailored Resume Output")
    
    if submit_btn:
        if not user_api_key or not jd_input or not resume_input:
            st.error("Please ensure all parameters are provided.")
        else:
            with st.spinner("Processing dynamic resume optimization pipeline..."):
                try:
                    meta, comprehensive_history = run_universal_agent(jd_input, resume_input, user_api_key)
                    
                    # Display Analytics Score
                    st.metric(label="Calculated ATS Match Rating", value=f"{meta.calculated_match_score}%")
                    st.markdown("---")
                    
                    st.markdown("### Technical Summary Profile")
                    st.info(f"**Professional Summary:**\n{meta.professional_summary}")
                    
                    st.markdown("### Technical Skills Core Matrix")
                    st.markdown(meta.technical_skills_matrix)
                    
                    st.markdown("### Detailed Chronological Experience Breakdown")
                    for job in comprehensive_history:
                        st.markdown(f"#### **{job['role_title']}** — *{job['company']}*")
                        for idx, bullet in enumerate(job["bullets"]):
                            st.markdown(f"**{idx+1}.** {bullet.optimized_text}")
                            st.caption(f"🔧 *Integrated Tags: {', '.join(bullet.keywords_integrated) if bullet.keywords_integrated else 'Verified Core Infrastructure'}*")
                    
                    unmapped_gaps = [g for g in meta.extracted_jd_keywords if g not in meta.verified_matching_skills]
                    if unmapped_gaps:
                        st.markdown("---")
                        st.markdown("### ⚠️ Blocked Tool Gaps (Sandbox Safety Filter)")
                        st.info(", ".join(unmapped_gaps))
                        
                except Exception as e:
                    st.error(f"An engine runtime error occurred: {str(e)}")
else:
    st.info("Provide parameters and click execute to process any technical profile universally without layout or title bias.")
