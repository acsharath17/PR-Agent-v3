import os
import re
import requests
import logging
from typing import TypedDict, List

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from redis import Redis
from rq import Queue
from rq.job import Job

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from tasks import analyze_pr_task

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)

# --- Load and validate environment ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key or not openai_api_key.startswith("sk-"):
    raise ValueError("Invalid or missing OPENAI_API_KEY")

# --- Flask ---
app = Flask(__name__)
CORS(app)

# --- Redis Queue ---
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
task_queue = Queue(connection=redis_conn)

@app.route("/analyze_pr", methods=["POST"])
def analyze_pr():
    data = request.get_json()
    if not data or "pr_url" not in data:
        return jsonify({"error": "Missing 'pr_url' in request body"}), 400

    job = task_queue.enqueue(analyze_pr_task, data["pr_url"])
    return jsonify({"job_id": job.get_id()}), 202

@app.route("/job_status/<job_id>", methods=["GET"])
def job_status(job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        if job.is_finished:
            return jsonify({"status": "finished", "result": job.result})
        elif job.is_failed:
            return jsonify({"status": "failed", "error": job.exc_info})
        else:
            return jsonify({"status": "in_progress"})
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == "__main__":
    app.run(debug=True)
