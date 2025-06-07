import os
import re
import requests
import logging
from typing import TypedDict, List

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)

class FileResult(TypedDict):
    filename: str
    diff: str
    is_apex: bool
    explanation: str
    review_comments: str
    business_summary: str

class PRState(TypedDict):
    pr_url: str
    diff: str
    files: List[FileResult]
    dev_summary: str
    business_summary: str

def call_llm_safe(prompt: str) -> str:
    try:
        return llm([HumanMessage(content=prompt)]).content.strip()
    except Exception as e:
        logging.error(f"OpenAI call failed: {e}")
        return f"Error: {str(e)}"

def is_apex_file(filename: str) -> bool:
    return filename.endswith(".cls") or filename.endswith(".trigger")

def fetch_pr_diff(state: PRState) -> PRState:
    diff = requests.get(state["pr_url"] + ".diff").text
    return {**state, "diff": diff}

def split_by_file(state: PRState) -> PRState:
    pattern = re.compile(r'diff --git a/(.*?) b/.*?\n(.*?)(?=\ndiff --git a/|\Z)', re.DOTALL)
    matches = pattern.findall(state["diff"])
    files: List[FileResult] = []
    for filename, file_diff in matches:
        files.append({
            "filename": filename.strip(),
            "diff": file_diff.strip(),
            "is_apex": is_apex_file(filename),
            "explanation": "",
            "review_comments": "",
            "business_summary": "",
        })
    return {**state, "files": files}

def process_files(state: PRState) -> PRState:
    updated_files = []
    for file in state["files"]:
        diff = file["diff"]
        explanation = call_llm_safe(f"Explain the changes:\n{diff}")
        review_comments = call_llm_safe(f"Review this Apex diff:\n{diff}") if file["is_apex"] else ""
        business_summary = call_llm_safe(f"Business summary of changes:\n{diff}")
        updated_files.append({**file, "explanation": explanation, "review_comments": review_comments, "business_summary": business_summary})
    return {**state, "files": updated_files}

def aggregate_summaries(state: PRState) -> PRState:
    explanations = [f"Changes in {f['filename']}:\n{f['explanation']}" for f in state["files"]]
    business_summaries = [f["business_summary"] for f in state["files"]]
    dev_summary = call_llm_safe("\n\n".join(explanations))
    business_summary = call_llm_safe("\n\n".join(business_summaries))
    return {**state, "dev_summary": dev_summary, "business_summary": business_summary}

# Build graph
builder = StateGraph(PRState)
builder.add_node("fetch_diff", fetch_pr_diff)
builder.add_node("split_by_file", split_by_file)
builder.add_node("process_files", process_files)
builder.add_node("aggregate_summaries", aggregate_summaries)
builder.set_entry_point("fetch_diff")
builder.add_edge("fetch_diff", "split_by_file")
builder.add_edge("split_by_file", "process_files")
builder.add_edge("process_files", "aggregate_summaries")
builder.add_edge("aggregate_summaries", END)
graph = builder.compile()

def analyze_pr_task(pr_url: str):
    result = graph.invoke({"pr_url": pr_url})
    return {
        "dev_summary": result["dev_summary"],
        "business_summary": result["business_summary"],
        "files": result["files"]
    }
