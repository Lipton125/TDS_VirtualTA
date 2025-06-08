import json
import os
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

# Config
DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in/"
AUTH_STATE_FILE = "auth.json"

def login_and_save_auth(playwright):
    """Manual login to Discourse and save session."""
    print("🔐 Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{DISCOURSE_BASE_URL}login")
    print("👉 Log in manually. Then click ▶ Resume in the Playwright debugger.")
    page.pause()
    context.storage_state(path=AUTH_STATE_FILE)
    browser.close()
    print("✅ Logged in and session saved to auth.json")

def expand_discourse_link(slugless_url: str):
    """Open a Discourse short link and return its expanded URL (after login)."""
    if not os.path.exists(AUTH_STATE_FILE):
        with sync_playwright() as p:
            login_and_save_auth(p)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=AUTH_STATE_FILE)
        page = context.new_page()
        print(f"🌐 Navigating to {slugless_url} ...")
        page.goto(slugless_url, wait_until="domcontentloaded")
        expanded_url = page.url
        print("✅ Expanded URL:", expanded_url)
        browser.close()
        return expanded_url

if __name__ == "__main__":
    # EXAMPLE: short URL
    short_url = "enter url"
    expanded = expand_discourse_link(short_url)
    print("\n🔗 Final Resolved URL:", expanded)
