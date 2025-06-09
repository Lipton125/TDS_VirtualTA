import os
import json
import sqlite3
import numpy as np
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import aiohttp
import asyncio
import re
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
import pytesseract

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY not set in environment")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    image: Optional[str] = None  # base64 encoded image

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

def extract_text_from_base64_image(image_base64: str) -> str:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.warning(f"Failed to extract text from image: {e}")
        return ""

def retrieve_chunks(query_emb: np.ndarray, top_k=MAX_RESULTS) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    chunks = []

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
    client_host = request.client.host
    logger.info(f"Incoming request from IP: {client_host}")
    logger.info(f"Request headers: {dict(request.headers)}")

    body_bytes = await request.body()
    try:
        body_str = body_bytes.decode('utf-8')
        body_json = json.loads(body_str)

        if "image" in body_json:
            img_len = len(body_json["image"])
            body_json["image_size_bytes"] = img_len
            del body_json["image"]

        logger.info(f"Request body: {json.dumps(body_json)}")
    except Exception as e:
        logger.warning(f"Failed to parse request body for logging: {e}")

    try:
        question_text = req.question

        if req.image:
            extracted_text = extract_text_from_base64_image(req.image)
            if extracted_text:
                logger.info(f"Extracted text from image: {extracted_text}")
                question_text += "\n\n" + extracted_text

        query_embedding = get_embedding(question_text)
        chunks = retrieve_chunks(query_embedding)

        if not chunks:
            return QueryResponse(answer="I couldn't find relevant content.", links=[])

        llm_output = await generate_llm_answer(question_text, chunks)

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

