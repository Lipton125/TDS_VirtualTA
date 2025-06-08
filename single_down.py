import os
import json
import requests
from datetime import datetime
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

# ========== CONFIGURATION ==========
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34
START_DATE = "2025-01-01"  # Inclusive
END_DATE = "2025-04-15"    # Inclusive
OUTPUT_DIR = "discourse_json"
AUTH_STATE_FILE = "auth.json"  # Playwright session storage
# ====================================

def login_and_save_auth(playwright):
    """Handle manual login and save session cookies."""
    print("🔐 Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{DISCOURSE_BASE_URL}/login")
    print("🌐 Please log in manually. Then press ▶️ (Resume) in Playwright's debug toolbar.")
    page.pause()
    context.storage_state(path=AUTH_STATE_FILE)
    browser.close()
    print("✅ Authentication state saved.")

def load_cookies_from_playwright():
    """Extract cookies from Playwright's auth state file."""
    if not os.path.exists(AUTH_STATE_FILE):
        with sync_playwright() as p:
            login_and_save_auth(p)
    
    with open(AUTH_STATE_FILE) as f:
        auth_state = json.load(f)
    return {c["name"]: c["value"] for c in auth_state["cookies"]}

def get_topic_ids(base_url, category_slug, category_id, start_date_str, end_date_str, cookies):
    """Fetch topic IDs within date range."""
    url = urljoin(base_url, f"c/{category_slug}/{category_id}.json")
    topic_ids = []
    page_num = 0
    consecutive_empty_pages = 0

    while consecutive_empty_pages < 5:  # Safety threshold
        paginated_url = f"{url}?page={page_num}"
        try:
            response = requests.get(paginated_url, cookies=cookies, timeout=30)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Failed to fetch page {page_num}: {e}")
            break

        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            consecutive_empty_pages += 1
            page_num += 1
            continue

        for topic in topics:
            created_at = topic.get("created_at")
            if created_at:
                created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                start_date = datetime.fromisoformat(start_date_str + "T00:00:00+00:00")
                end_date = datetime.fromisoformat(end_date_str + "T23:59:59+00:00")
                if start_date <= created_date <= end_date:
                    topic_ids.append(topic["id"])

        page_num += 1

    return list(set(topic_ids))  # Deduplicate

def download_topic(topic_id, base_url, cookies, output_dir):
    """Download full JSON of a single topic and save it."""
    topic_url = urljoin(base_url, f"t/{topic_id}.json")
    try:
        response = requests.get(topic_url, cookies=cookies, timeout=30)
        response.raise_for_status()
        topic_data = response.json()
    except Exception as e:
        print(f"❌ Failed to fetch topic {topic_id}: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"topic_{topic_id}.json")
    with open(output_path, "w") as f:
        json.dump(topic_data, f, indent=2)
    print(f"✅ Saved topic {topic_id} to {output_path}")

def main():
    # Authenticate and load cookies
    cookies = load_cookies_from_playwright()
    
    # ---- Download specific topic ----
    topic_id = 164205
    download_topic(topic_id, DISCOURSE_BASE_URL, cookies, OUTPUT_DIR)

    # ---- Optional: Fetch topic IDs in range ----
    # topic_ids = get_topic_ids(
    #     DISCOURSE_BASE_URL, CATEGORY_SLUG, CATEGORY_ID,
    #     START_DATE, END_DATE, cookies
    # )
    # print(f"Found {len(topic_ids)} topics in date range.")
    # for tid in topic_ids:
    #     download_topic(tid, DISCOURSE_BASE_URL, cookies, OUTPUT_DIR)

if __name__ == "__main__":
    main()

