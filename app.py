import os
import re
import requests
from typing import TypedDict, List

from flask import Flask, request, jsonify
from flask_cors import CORS

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI  # Updated import
from langchain.schema import HumanMessage

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment variables")

# --- Initialize Flask app ---
app = Flask(__name__)
CORS(app)

# --- Initialize LangChain OpenAI wrapper ---
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0,
    openai_api_key=openai_api_key
)

# --- Define state schemas ---
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

# --- Helpers ---
def is_apex_file(filename: str) -> bool:
    return filename.endswith(".cls") or filename.endswith(".trigger")

# --- LangGraph Nodes ---
def fetch_pr_diff(state: PRState) -> PRState:
    pr_url = state["pr_url"]
    diff_url = pr_url + ".diff"
    diff = requests.get(diff_url).text
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

        explanation = llm([
            HumanMessage(content=f"Explain in detail the changes made in this file:\n\n{diff}")
        ]).content.strip()

        review_comments = ""
        if file["is_apex"]:
            review_comments = llm([
                HumanMessage(content=f"Review this Apex code diff:\n\n{diff}")
            ]).content.strip()

        business_summary = llm([
            HumanMessage(content=f"Summarize this for business users (no technical terms):\n\n{diff}")
        ]).content.strip()

        updated_files.append({
            **file,
            "explanation": explanation,
            "review_comments": review_comments,
            "business_summary": business_summary,
        })

    return {**state, "files": updated_files}

def aggregate_summaries(state: PRState) -> PRState:
    explanations = [f"Changes in {f['filename']}:\n{f['explanation']}" for f in state["files"]]
    business_summaries = [f["business_summary"] for f in state["files"]]

    dev_summary = llm([
        HumanMessage(content="\n\n".join(explanations))
    ]).content.strip()

    business_summary = llm([
        HumanMessage(content="\n\n".join(business_summaries))
    ]).content.strip()

    return {
        **state,
        "dev_summary": dev_summary,
        "business_summary": business_summary,
    }

# --- Build LangGraph ---
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

# --- Flask Endpoints ---
@app.route("/analyze_pr", methods=["POST"])
def analyze_pr():
    data = request.get_json()
    if not data or "pr_url" not in data:
        return jsonify({"error": "Missing 'pr_url' in request body"}), 400

    pr_url = data["pr_url"]

    try:
        result = graph.invoke({"pr_url": pr_url})
        return jsonify({
            "dev_summary": result["dev_summary"],
            "business_summary": result["business_summary"],
            "files": result["files"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def hello_world():
    return 'Hello, World!'

# --- Local testing ---
if __name__ == "__main__":
    app.run(debug=True)
