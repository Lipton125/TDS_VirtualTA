import os
import json
import sqlite3
import uuid
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from link import expand_discourse_link  # Ensure link.py is present

# === Config ===
FORUM_DIR = "downloaded_threads"
COURSE_DIR = "markdown_files"
DB_PATH = "knowledge_base.db"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 70

EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")

# === URL Cache ===
URL_CACHE_PATH = "url_cache.json"
url_cache = {}

if os.path.exists(URL_CACHE_PATH):
    with open(URL_CACHE_PATH, "r") as f:
        url_cache = json.load(f)

# === Helpers ===

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return chunks

def embed(texts):
    return EMBEDDING_MODEL.encode(texts).tolist()

def create_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS forum_chunks (
            chunk_id TEXT PRIMARY KEY,
            post_id INTEGER,
            post_number INTEGER,
            topic_id INTEGER,
            topic_title TEXT,
            author TEXT,
            url TEXT,
            text TEXT,
            embedding BLOB
        )
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS course_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_file TEXT,
            section_title TEXT,
            url TEXT,
            text TEXT,
            embedding BLOB
        )
    ''')
    conn.commit()
    return conn

# === Processors ===

def process_forum_json(filepath, conn):
    global url_cache

    with open(filepath, 'r', encoding='utf-8') as f:
        posts = json.load(f)

    if not posts:
        return

    topic_id = posts[0]['topic_id']
    topic_base_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}/1"

    # Expand only once per topic
    if topic_base_url in url_cache:
        expanded_base_url = url_cache[topic_base_url]
    else:
        try:
            expanded_full_url = expand_discourse_link(topic_base_url)
            expanded_base_url = expanded_full_url.rsplit('/', 1)[0]
            url_cache[topic_base_url] = expanded_base_url
        except Exception as e:
            print(f"❌ Failed to expand topic {topic_id}: {e}")
            expanded_base_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}"

    for post in posts:
        chunks = chunk_text(post["content"])
        embeddings = embed(chunks)

        full_url = f"{expanded_base_url}/{post['post_number']}"

        for chunk, emb in zip(chunks, embeddings):
            conn.execute(
                '''INSERT INTO forum_chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    str(uuid.uuid4()),
                    post["post_id"],
                    post["post_number"],
                    post["topic_id"],
                    post["topic_title"],
                    post["author"],
                    full_url,
                    chunk,
                    json.dumps(emb)
                )
            )

def process_course_md(filepath, conn):
    content = Path(filepath).read_text(encoding='utf-8')
    lines = content.splitlines()

    # Extract front matter
    url = None
    if lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip().startswith("original_url:"):
                url = lines[i].split(":", 1)[1].strip().strip('"')
            elif lines[i].strip() == "---":
                lines = lines[i+1:]  # Remove front matter
                break

    section_title = ""
    buffer = ""
    for line in lines:
        if line.strip().startswith("#"):
            section_title = line.strip("# ").strip()
            if buffer:
                chunks = chunk_text(buffer)
                embeddings = embed(chunks)
                for chunk, emb in zip(chunks, embeddings):
                    conn.execute(
                        '''INSERT INTO course_chunks VALUES (?, ?, ?, ?, ?, ?)''',
                        (
                            str(uuid.uuid4()),
                            os.path.basename(filepath),
                            section_title,
                            url,
                            chunk,
                            json.dumps(emb)
                        )
                    )
                buffer = ""
        else:
            buffer += line + "\n"
    if buffer:
        chunks = chunk_text(buffer)
        embeddings = embed(chunks)
        for chunk, emb in zip(chunks, embeddings):
            conn.execute(
                '''INSERT INTO course_chunks VALUES (?, ?, ?, ?, ?, ?)''',
                (
                    str(uuid.uuid4()),
                    os.path.basename(filepath),
                    section_title,
                    url,
                    chunk,
                    json.dumps(emb)
                )
            )

# === Main ===

def main():
    conn = create_db()

    # Process forum files
    for file in os.listdir(FORUM_DIR):
        if file.endswith(".json"):
            process_forum_json(os.path.join(FORUM_DIR, file), conn)

    # Process course markdown files
    for file in os.listdir(COURSE_DIR):
        if file.endswith(".md"):
            process_course_md(os.path.join(COURSE_DIR, file), conn)

    conn.commit()
    conn.close()

    # Save updated URL cache
    with open(URL_CACHE_PATH, "w") as f:
        json.dump(url_cache, f, indent=2)

    print(f"✅ Knowledge base built and stored in {DB_PATH}")

if __name__ == "__main__":
    main()

