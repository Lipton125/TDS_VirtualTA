import os
import json
import sqlite3
import numpy as np
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import aiohttp
import asyncio
import re
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set in environment")

app = FastAPI()

# Add CORS middleware to handle preflight requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["http://localhost:3000"] or your frontend URL for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.5
MAX_RESULTS = 10

class QueryRequest(BaseModel):
    question: str

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_embedding(text: str) -> np.ndarray:
    return EMBEDDING_MODEL.encode(text)

def retrieve_chunks(query_emb: np.ndarray, top_k=MAX_RESULTS) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    chunks = []

    # forum_chunks
    cur.execute("SELECT chunk_id, topic_id, topic_title, url, text, embedding FROM forum_chunks")
    for row in cur.fetchall():
        chunk_id, topic_id, topic_title, url, text, emb_json = row
        emb = np.array(json.loads(emb_json))
        sim = cosine_similarity(query_emb, emb)
        if sim >= SIMILARITY_THRESHOLD:
            chunks.append({
                "source": "forum",
                "chunk_id": chunk_id,
                "title": topic_title,
                "url": url,
                "text": text,
                "similarity": sim
            })

    # course_chunks
    cur.execute("SELECT chunk_id, source_file, section_title, url, text, embedding FROM course_chunks")
    for row in cur.fetchall():
        chunk_id, source_file, section_title, url, text, emb_json = row
        emb = np.array(json.loads(emb_json))
        sim = cosine_similarity(query_emb, emb)
        if sim >= SIMILARITY_THRESHOLD:
            chunks.append({
                "source": "course",
                "chunk_id": chunk_id,
                "title": section_title or source_file,
                "url": url,
                "text": text,
                "similarity": sim
            })

    conn.close()
    chunks.sort(key=lambda x: x["similarity"], reverse=True)
    return chunks[:top_k]

async def generate_llm_answer(question: str, chunks: List[Dict]) -> str:
    context = ""
    for chunk in chunks:
        source_type = "Forum post" if chunk["source"] == "forum" else "Course material"
        context += f"{source_type} (URL: {chunk['url']}):{chunk['text'][:1500]}"

    prompt = f"""
You are not a conversational assistant. You are a compliance checker.

DO NOT answer based on general knowledge or what the user wants to hear.

ONLY answer based on the course materials provided below. Do NOT infer or inject anything. If the material does not clearly answer the question, reply with:

"I don't have enough information to answer this question."

---

Context:
{context}

---

Question:
{question}

Based on ONLY the context above, what does the course recommend?

Format your response like this:

1. [Factual answer only]

Sources:
1. URL: [exact URL], Text: [short quote from that page]
2. URL: [exact URL], Text: [short quote from that page]
"""

    headers = {
        "Authorization": API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1
    }

    async with aiohttp.ClientSession() as session:
        async with session.post("https://aipipe.org/openai/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=await response.text())
            result = await response.json()
            return result["choices"][0]["message"]["content"]

@app.post("/query", response_model=QueryResponse)
async def query_virtual_ta(req: QueryRequest, request: Request):
    # Log client IP
    client_host = request.client.host
    logger.info(f"Incoming request from IP: {client_host}")

    # Log request headers
    headers_dict = dict(request.headers)
    logger.info(f"Request headers: {headers_dict}")

    # Log raw body (request JSON)
    body_bytes = await request.body()
    logger.info(f"Request raw body: {body_bytes.decode('utf-8')}")

    try:
        query_embedding = get_embedding(req.question)
        chunks = retrieve_chunks(query_embedding)

        if not chunks:
            return QueryResponse(answer="I couldn't find relevant content.", links=[])

        llm_output = await generate_llm_answer(req.question, chunks)

        # Extract main answer and links from "Sources:" section
        if "Sources:" in llm_output:
            answer_part, sources_part = llm_output.split("Sources:", 1)
        else:
            answer_part = llm_output
            sources_part = ""

        links = []
        for match in re.finditer(r"URL:\s*(\S+),\s*Text:\s*(.*)", sources_part):
            url, text = match.groups()
            links.append(Link(url=url.strip(), text=text.strip()))

        return QueryResponse(answer=answer_part.strip(), links=links)

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

