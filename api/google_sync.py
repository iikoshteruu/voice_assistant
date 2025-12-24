"""Google API sync for Calendar, Gmail, and Sheets."""
import base64
import json
import logging
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/calendar",  # Full calendar access (read/write)
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/contacts.readonly",
]

CREDENTIALS_PATH = "/data/google_credentials.json"
TOKEN_PATH = "/data/google_token.json"
CONTACTS_PATH = "/data/contacts.json"


def get_credentials() -> Optional[Credentials]:
    """Get or refresh Google credentials."""
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_token(creds)
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            creds = None

    return creds


def save_token(creds: Credentials):
    """Save token to file."""
    with open(TOKEN_PATH, "w") as f:
        f.write(creds.to_json())


def run_local_auth() -> bool:
    """Run OAuth flow with local server (must be run interactively)."""
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError("Google credentials not configured")

    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        creds = flow.run_local_server(port=8089, open_browser=False)
        save_token(creds)
        return True
    except Exception as e:
        logger.error(f"Auth failed: {e}")
        return False


def get_auth_url(redirect_uri: str) -> tuple[str, str]:
    """Get OAuth authorization URL for manual flow."""
    if not os.path.exists(CREDENTIALS_PATH):
        raise FileNotFoundError("Google credentials not configured")

    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    flow.redirect_uri = redirect_uri

    auth_url, state = flow.authorization_url(
        prompt="consent",
        access_type="offline"
    )
    return auth_url, state


def complete_auth(code: str, redirect_uri: str) -> bool:
    """Complete OAuth flow with authorization code."""
    try:
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
        flow.redirect_uri = redirect_uri
        flow.fetch_token(code=code)
        save_token(flow.credentials)
        return True
    except Exception as e:
        logger.error(f"Auth failed: {e}")
        return False


def is_authenticated() -> bool:
    """Check if we have valid credentials."""
    creds = get_credentials()
    return creds is not None and creds.valid


def create_calendar_event(summary: str, start_time: str, end_time: str = None, description: str = "") -> dict:
    """Create a calendar event.

    Args:
        summary: Event title
        start_time: ISO format datetime (e.g., "2025-12-26T14:00:00")
        end_time: ISO format datetime (optional, defaults to 1 hour after start)
        description: Event description
    """
    creds = get_credentials()
    if not creds:
        return {"error": "Not authenticated"}

    try:
        service = build("calendar", "v3", credentials=creds)

        # Parse start time and create end time if not provided
        from datetime import datetime, timedelta
        start_dt = datetime.fromisoformat(start_time.replace("Z", ""))
        if not end_time:
            end_dt = start_dt + timedelta(hours=1)
            end_time = end_dt.isoformat()

        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start_time, "timeZone": "Asia/Tokyo"},
            "end": {"dateTime": end_time, "timeZone": "Asia/Tokyo"},
        }

        created = service.events().insert(calendarId="primary", body=event).execute()
        return {"success": True, "event_id": created["id"], "link": created.get("htmlLink")}

    except Exception as e:
        logger.error(f"Calendar event creation failed: {e}")
        return {"error": str(e)}


async def sync_calendar(add_to_qdrant_func, days_ahead: int = 14) -> dict:
    """Sync calendar events to Qdrant."""
    creds = get_credentials()
    if not creds:
        return {"error": "Not authenticated"}

    try:
        service = build("calendar", "v3", credentials=creds)

        now = datetime.utcnow()
        time_min = now.isoformat() + "Z"
        time_max = (now + timedelta(days=days_ahead)).isoformat() + "Z"

        events_result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            maxResults=50,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        synced = 0

        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "No title")
            description = event.get("description", "")
            location = event.get("location", "")

            content = f"Calendar event on {start}: {summary}"
            if location:
                content += f" at {location}"
            if description:
                content += f". Details: {description}"

            await add_to_qdrant_func(
                content=content,
                source="google_calendar",
                metadata={"event_id": event["id"], "start": start}
            )
            synced += 1

        return {"synced": synced, "source": "calendar"}

    except Exception as e:
        logger.error(f"Calendar sync failed: {e}")
        return {"error": str(e)}


async def sync_gmail(add_to_qdrant_func, max_emails: int = 200) -> dict:
    """Sync recent emails to Qdrant."""
    creds = get_credentials()
    if not creds:
        return {"error": "Not authenticated"}

    try:
        service = build("gmail", "v1", credentials=creds)

        results = service.users().messages().list(
            userId="me",
            maxResults=max_emails,
            q="is:inbox"
        ).execute()

        messages = results.get("messages", [])
        synced = 0

        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()

            headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
            snippet = msg.get("snippet", "")

            content = f"Email from {headers.get('From', 'Unknown')} on {headers.get('Date', 'Unknown')}: {headers.get('Subject', 'No subject')}. {snippet}"

            await add_to_qdrant_func(
                content=content,
                source="gmail",
                metadata={"message_id": msg_ref["id"]}
            )
            synced += 1

        return {"synced": synced, "source": "gmail"}

    except Exception as e:
        logger.error(f"Gmail sync failed: {e}")
        return {"error": str(e)}


async def sync_sheet(add_to_qdrant_func, spreadsheet_id: str, range_name: str = "A1:Z100") -> dict:
    """Sync a Google Sheet to Qdrant."""
    creds = get_credentials()
    if not creds:
        return {"error": "Not authenticated"}

    try:
        service = build("sheets", "v4", credentials=creds)

        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()

        values = result.get("values", [])
        if not values:
            return {"synced": 0, "source": "sheets"}

        # Use first row as headers
        headers = values[0] if values else []
        synced = 0

        for row in values[1:]:
            row_data = dict(zip(headers, row))
            content = f"Spreadsheet data: {json.dumps(row_data)}"

            await add_to_qdrant_func(
                content=content,
                source="google_sheets",
                metadata={"spreadsheet_id": spreadsheet_id}
            )
            synced += 1

        return {"synced": synced, "source": "sheets", "spreadsheet_id": spreadsheet_id}

    except Exception as e:
        logger.error(f"Sheets sync failed: {e}")
        return {"error": str(e)}


def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email via Gmail API."""
    creds = get_credentials()
    if not creds:
        return {"error": "Not authenticated"}

    try:
        service = build("gmail", "v1", credentials=creds)

        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()

        return {"success": True, "message_id": sent["id"]}

    except Exception as e:
        logger.error(f"Email send failed: {e}")
        return {"error": str(e)}


def sync_contacts() -> dict:
    """Sync Google Contacts to local file."""
    creds = get_credentials()
    if not creds:
        return {"error": "Not authenticated"}

    try:
        service = build("people", "v1", credentials=creds)

        contacts = {}
        next_page_token = None

        while True:
            results = service.people().connections().list(
                resourceName="people/me",
                pageSize=200,
                personFields="names,emailAddresses",
                pageToken=next_page_token
            ).execute()

            for person in results.get("connections", []):
                names = person.get("names", [])
                emails = person.get("emailAddresses", [])

                if names and emails:
                    # Get primary or first name
                    name = names[0].get("displayName", "").lower()
                    first_name = names[0].get("givenName", "").lower()
                    email = emails[0].get("value", "")

                    if name and email:
                        contacts[name] = email
                    if first_name and email and first_name != name:
                        contacts[first_name] = email

            next_page_token = results.get("nextPageToken")
            if not next_page_token:
                break

        # Save to file
        with open(CONTACTS_PATH, "w") as f:
            json.dump(contacts, f)

        return {"synced": len(contacts), "source": "contacts"}

    except Exception as e:
        logger.error(f"Contacts sync failed: {e}")
        return {"error": str(e)}


def lookup_contact(name: str) -> Optional[str]:
    """Look up email by contact name."""
    try:
        if os.path.exists(CONTACTS_PATH):
            with open(CONTACTS_PATH, "r") as f:
                contacts = json.load(f)

            name_lower = name.lower().strip()

            # Exact match
            if name_lower in contacts:
                return contacts[name_lower]

            # Partial match
            for contact_name, email in contacts.items():
                if name_lower in contact_name or contact_name in name_lower:
                    return email

        return None
    except Exception as e:
        logger.error(f"Contact lookup failed: {e}")
        return None


def get_all_contacts() -> dict:
    """Get all synced contacts."""
    try:
        if os.path.exists(CONTACTS_PATH):
            with open(CONTACTS_PATH, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}
