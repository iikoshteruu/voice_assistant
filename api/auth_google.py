#!/usr/bin/env python3
"""One-time Google OAuth script. Run this to authenticate."""
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
]

CREDENTIALS_PATH = "/data/google_credentials.json"
TOKEN_PATH = "/data/google_token.json"

def main():
    if not os.path.exists(CREDENTIALS_PATH):
        print(f"Error: {CREDENTIALS_PATH} not found")
        print("Copy your Google OAuth credentials JSON to that path first.")
        sys.exit(1)

    print("Starting Google OAuth flow...")
    print("A browser window will open. If not, copy the URL shown below.\n")

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(
        port=8089,
        prompt="consent",
        open_browser=False
    )

    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())

    print(f"\nSuccess! Token saved to {TOKEN_PATH}")
    print("You can now sync your Google data with Socrates.")

if __name__ == "__main__":
    main()
