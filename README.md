# TDS Virtual Teaching Assistant Project

## Overview

The **TDS Virtual Teaching Assistant (Virtual TA)** is an AI-driven question-answering system built with FastAPI. It provides precise, source-based answers by leveraging a curated knowledge base of course materials and forum discussions related to the TDS Jan 2025 course.

### Key Features and Workflow

- **Multi-Source Knowledge Base**  
  The system combines indexed course content and forum posts stored in an SQLite database, with pre-computed sentence embeddings for efficient semantic search.

- **Semantic Search with Embeddings**  
  Incoming questions are converted into embeddings using a SentenceTransformer model (`BAAI/bge-base-en-v1.5`). The system retrieves the most relevant content chunks from both course and forum data by computing cosine similarity against stored embeddings, filtering based on a configurable similarity threshold.

- **Image Text Extraction (OCR)**  
  The API supports optional image inputs encoded in base64. It extracts text from these images using Tesseract OCR (`pytesseract`), appending any extracted text to the original question to improve context and answer accuracy.

- **Contextual LLM Answer Generation**  
  The retrieved text chunks form a focused context prompt for an external Large Language Model (LLM) API (`gpt-4o-mini` via aipipe.org). The LLM is instructed to provide factual answers strictly based on the supplied course/forum context, including exact URLs and textual citations to ensure compliance and transparency.

- **Structured Response with Sources**  
  The API response contains a clear, factual answer and a list of source links with short quotes, allowing users to verify information and explore the original material.

- **Robust Logging and Error Handling**  
  Incoming requests are logged with client IP and headers. The system handles errors in OCR, database access, and external API calls, returning meaningful HTTP errors when necessary.

This architecture ensures that student questions are answered with high accuracy and transparency, grounded firmly in authorized course materials and community discussions.

---



---

## Project Structure

### 1. Knowledge Base
The knowledge base combines two primary data sources to power the Virtual TA:

- **Course Content:**  
  Complete course content for TDS Jan 2025, scraped and updated as of April 15, 2025.

- **Discourse Forum Posts:**  
  Posts from the TDS Discourse forum covering the period January 1, 2025, to April 14, 2025.

---

### 2. Data Scraping Tools

- **Discourse Posts Scraping:**
  - `discourse_by_date_range.py`  
    Scrapes forum posts within a specified date range.
  - `discourse_by_post_id.py`  
    Scrapes specific posts outside the normal date ranges, identified by post ID.

- **Course Content Scraping:**
  - `website_downloader_full.py`  
    Downloads and parses the entire course website content.

---

### 3. Knowledge Base Creation

- `base_creation_test.py`  
  Processes and consolidates the scraped data into a structured knowledge base ready for querying.

---

### 4. API Implementation

Two Python API scripts provide endpoints for querying the Virtual TA:

- `virtual_ta_api_test.py`  
  Provides a REST API for question answering based on the knowledge base.

- `virtual_ta_api_test_img.py`  
  Extended API with support for image attachments (e.g., base64-encoded screenshots) in questions.

---

## API

- **Python Version:** 3.10.17
- **API Endpoint:**  
  [https://tds-virtual-ta.serveo.net/query](https://tds-virtual-ta.serveo.net/query)

---

## How to Use

1. **Run the scraping scripts** to keep the knowledge base up-to-date.
2. **Build the knowledge base** using `base_creation_test.py`.
3. **Deploy the API** (`virtual_ta_api_test.py` or `virtual_ta_api_test_img.py`).
4. **Send POST requests** to the API endpoint with student questions and optional images to receive answers.

---

## Example API Requests

You can query the Virtual TA API using a `POST` request with JSON data. The request can include a `question` string and optionally an image encoded in base64.

```bash
curl "https://tds-virtual-ta.serveo.net/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Enter your question here.\", \"image\": \"$(base64 -w0 /img_path/img [Optional].)\"}"
