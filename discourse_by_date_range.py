import os
import json
import requests
from datetime import datetime, timezone
from urllib.parse import urljoin, urlencode
from playwright.sync_api import sync_playwright

# Auth & Config.

DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34
START_DATE = "2025-01-01"  # Inclusive
END_DATE = "2025-04-15"    # Inclusive
OUTPUT_DIR = "discourse_json"
AUTH_STATE_FILE = "auth.json"
POST_ID_BATCH_SIZE = 50
MAX_CONSECUTIVE_PAGES_WITHOUT_NEW_TOPICS = 5

# Login in case of absence of auth.json.
def login_and_save_auth(playwright):
    """Launch browser for login and save cookies."""
    print(" Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{DISCOURSE_BASE_URL}/login")
    print(" Please log in manually. Then press (Resume) in the debug bar.")
    page.pause()
    context.storage_state(path=AUTH_STATE_FILE)
    browser.close()
    print(" Login session saved.")

def load_cookies_from_playwright():
    """Load cookies from Playwright session."""
    if not os.path.exists(AUTH_STATE_FILE):
        with sync_playwright() as p:
            login_and_save_auth(p)

    with open(AUTH_STATE_FILE) as f:
        auth_state = json.load(f)
    return {c["name"]: c["value"] for c in auth_state["cookies"]}

# Fetch and save.

def get_topic_ids(base_url, category_slug, category_id, start_date_str, end_date_str, cookies):
    url = urljoin(base_url, f"c/{category_slug}/{category_id}.json")
    topic_ids = []
    page = 0

    start_dt_naive = datetime.fromisoformat(start_date_str + "T00:00:00")
    start_dt = start_dt_naive.replace(tzinfo=timezone.utc)
    end_dt_naive = datetime.fromisoformat(end_date_str + "T23:59:59.999999")
    end_dt = end_dt_naive.replace(tzinfo=timezone.utc)

    print(f"Fetching topic IDs between {start_dt} and {end_dt}...")

    consecutive_pages_with_no_new_unique_topics = 0
    last_known_unique_topic_count = 0

    while True:
        paginated_url = f"{url}?page={page}"
        try:
            response = requests.get(paginated_url, cookies=cookies, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page {page}: {e}")
            break

        try:
            data = response.json()
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from page {page}. Content: {response.text[:200]}...")
            break

        topics_on_page = data.get("topic_list", {}).get("topics", [])

        if not topics_on_page:
            print(f"No more topics on page {page}.")
            break

        count_before = len(set(topic_ids))

        for topic in topics_on_page:
            created_at_str = topic.get("created_at")
            if created_at_str:
                try:
                    created_date = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                except ValueError:
                    continue

                if start_dt <= created_date <= end_dt:
                    topic_ids.append(topic["id"])

        current_count = len(set(topic_ids))

        if current_count == count_before:
            consecutive_pages_with_no_new_unique_topics += 1
            print(f"Page {page} yielded no new topics. Consecutive stale pages: {consecutive_pages_with_no_new_unique_topics}")
        else:
            consecutive_pages_with_no_new_unique_topics = 0

        last_known_unique_topic_count = current_count

        if consecutive_pages_with_no_new_unique_topics >= MAX_CONSECUTIVE_PAGES_WITHOUT_NEW_TOPICS:
            print("Too many stale pages. Stopping.")
            break

        if not data.get("topic_list", {}).get("more_topics_url"):
            print("No more_topics_url. End of pages.")
            break

        print(f"Page {page} OK. Continuing...")
        page += 1

    return list(set(topic_ids))

def get_full_topic_json(base_url, topic_id, cookies):
    topic_url = urljoin(base_url, f"t/{topic_id}.json")
    print(f"Fetching topic {topic_id}")

    try:
        response = requests.get(topic_url, cookies=cookies, timeout=30)
        response.raise_for_status()
        topic_data = response.json()
    except Exception as e:
        print(f"Failed to fetch topic {topic_id}: {e}")
        return None

    post_stream = topic_data.get("post_stream")
    if not post_stream or "stream" not in post_stream or "posts" not in post_stream:
        return topic_data

    all_ids = post_stream.get("stream", [])
    loaded_ids = {post["id"] for post in post_stream.get("posts", [])}
    missing_ids = [pid for pid in all_ids if pid not in loaded_ids]

    fetched_posts = []
    for i in range(0, len(missing_ids), POST_ID_BATCH_SIZE):
        batch_ids = missing_ids[i:i + POST_ID_BATCH_SIZE]
        query = [("post_ids[]", pid) for pid in batch_ids]
        posts_url = urljoin(base_url, f"t/{topic_id}/posts.json")

        try:
            batch_response = requests.get(posts_url, params=query, cookies=cookies, timeout=30)
            batch_response.raise_for_status()
            batch_data = batch_response.json()
            if isinstance(batch_data, list):
                fetched_posts.extend(batch_data)
            elif "post_stream" in batch_data and "posts" in batch_data["post_stream"]:
                fetched_posts.extend(batch_data["post_stream"]["posts"])
            elif "posts" in batch_data:
                fetched_posts.extend(batch_data["posts"])
        except Exception as e:
            print(f"Error fetching post batch for topic {topic_id}: {e}")

    if fetched_posts:
        existing = {p['id']: p for p in topic_data["post_stream"]["posts"]}
        for p in fetched_posts:
            if p['id'] not in existing:
                topic_data["post_stream"]["posts"].append(p)
                existing[p['id']] = p

        full_posts = [existing[pid] for pid in all_ids if pid in existing]
        topic_data["post_stream"]["posts"] = full_posts

    return topic_data

def save_topic_json(topic_id, data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"topic_{topic_id}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"Error saving topic {topic_id}: {e}")

def main():
    print("Starting Discourse Downloader")
    cookies = load_cookies_from_playwright()

    topic_ids = get_topic_ids(
        DISCOURSE_BASE_URL,
        CATEGORY_SLUG,
        CATEGORY_ID,
        START_DATE,
        END_DATE,
        cookies
    )

    if not topic_ids:
        print("No topics found. Exiting.")
        return

    print(f"\n Downloading {len(topic_ids)} topics...\n")
    successes = 0
    failed = []

    for i, tid in enumerate(topic_ids, 1):
        print(f"\n [{i}/{len(topic_ids)}] Topic ID: {tid}")
        data = get_full_topic_json(DISCOURSE_BASE_URL, tid, cookies)
        if data:
            save_topic_json(tid, data, OUTPUT_DIR)
            successes += 1
        else:
            failed.append(tid)

    print("\n DONE")
    print(f"Downloaded: {successes}")
    print(f"Failed: {len(failed)} → {failed if failed else 'None'}")
    print(f" Output saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()

