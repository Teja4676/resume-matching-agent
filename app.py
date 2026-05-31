import streamlit as st
import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# --- CHANGE THIS IMPORT ---
from langchain_groq import ChatGroq 

# (Keep your Pydantic schemas exactly the same as they were before)

# --- CHANGER YOUR CORE AGENT ENGINE TO USE GROQ ---
def run_agent(job_description: str, raw_resume: str, api_key: str) -> FullOptimizedResume:
    # Swapping OpenAI for Groq's high-speed free model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        temperature=0.0, 
        groq_api_key=api_key
    )
    
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

# --- UPDATE THE TEXT BOX LABEL ---
with col1:
    st.subheader("1. Input Configuration")
    user_api_key = st.text_input("Enter your Free Groq API Key", type="password")
    # (Keep the rest of your Streamlit UI components exactly as they were...)
